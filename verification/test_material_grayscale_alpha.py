"""Preserve original grayscale/alpha integer data without premultiplication."""
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_dataset as dataset
from train_material_height import load_pair


def independent_la_png(path, values, gamma=0.45455):
    height, width, channels = values.shape
    assert channels == 2
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    rows = b"".join(b"\0" + row.astype(values.dtype.newbyteorder(">"), copy=False).tobytes() for row in values)
    header = struct.pack(">IIBBBBB", width, height, values.dtype.itemsize * 8, 4, 0, 0, 0)
    path.write_bytes(dataset.PNG_SIGNATURE + chunk(b"IHDR", header)
        + chunk(b"gAMA", struct.pack(">I", round(gamma * 100000)))
        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
def test_independently_encoded_two_channels_round_trip_exactly(tmp_path, dtype):
    maximum = np.iinfo(dtype).max
    gray = np.resize(np.array([0, 1, maximum // 2, maximum // 2 + 1, maximum], dtype=dtype), (5, 7))
    alpha = np.resize(np.array([0, 1, maximum - 4, maximum - 2, maximum], dtype=dtype), (5, 7))
    original = np.stack((gray, alpha), axis=-1)
    source = tmp_path / "original.png"
    independent_la_png(source, original)
    before = source.read_bytes()
    decoded, metadata = dataset.read_png(source)
    np.testing.assert_array_equal(decoded, original)
    assert decoded.dtype == dtype and metadata["channels"] == 2 and metadata["png_color_type"] == 4
    target = tmp_path / "crop.png"
    crop = decoded[1:4, 2:6]
    dataset.write_png(target, crop, metadata)
    result, header = dataset.read_png(target)
    np.testing.assert_array_equal(result, original[1:4, 2:6])
    assert header["sample_bits"] == np.dtype(dtype).itemsize * 8
    assert header["png_gamma"] == metadata["png_gamma"]
    assert header["channels"] == 2 and header["png_color_type"] == 4
    assert source.read_bytes() == before
    # Independent scanline inspection checks byte order and channel preservation.
    payload = b"".join(value for kind, value in dataset.png_chunks(target.read_bytes()) if kind == b"IDAT")
    raw = zlib.decompress(payload)
    stride = crop.shape[1] * 2 * np.dtype(dtype).itemsize
    for row in range(crop.shape[0]):
        assert raw[row * (stride + 1)] == 0
        values = np.frombuffer(raw[row * (stride + 1) + 1:(row + 1) * (stride + 1)], dtype=np.dtype(dtype).newbyteorder(">"))
        np.testing.assert_array_equal(values.reshape(crop.shape[1], 2), crop[row])


def material(tmp_path, alpha=65531):
    sources = tmp_path / "sources"
    folder = sources / "surface"
    folder.mkdir(parents=True)
    gray = np.resize(np.array([32767, 32768, 32769], dtype=np.uint16), (32, 32))
    height = np.stack((gray, np.full_like(gray, alpha)), axis=-1)
    independent_la_png(folder / "surface_disp_4k.png", height)
    for suffix, channels in (("diff", 3), ("nor_gl", 3), ("rough", 1)):
        dataset.write_png(folder / f"surface_{suffix}_4k.png", np.full((32, 32, channels), 32768, dtype=np.uint16))
    materials = dataset.discover_materials(sources, 16)
    assert len(materials) == 1 and materials[0]["ready"]
    return folder, height, materials[0]


def test_prepared_height_retains_alpha_and_adjacent_gray_codes_in_training(tmp_path):
    folder, original, source = material(tmp_path)
    original_hashes = {p.name: dataset.file_sha256(p) for p in folder.iterdir()}
    output = tmp_path / "dataset"
    records = dataset.prepare_material(source, output, {}, True, .2)
    record = records[0]
    directory = output / "samples" / record["sample_id"]
    assert dataset.verify_sample(directory, check_sources=True) == []
    codes, _ = dataset.read_png(directory / record["maps"]["height"])
    x, y, width, height = record["crop_rectangle_top_left_xywh"]
    np.testing.assert_array_equal(codes, original[y:y + height, x:x + width])
    sample = {"metadata": record, "metadata_path": directory / "sample.json",
        "input_path": directory / record["maps"]["input"], "height_path": directory / record["maps"]["height"]}
    _, target = load_pair(sample, torch.device("cpu"))
    np.testing.assert_array_equal(target.numpy()[0, 0], codes[..., 0].astype(np.float32) / np.float32(65535))
    assert torch.unique(target).numel() == 3
    assert record["map_metadata"]["height"]["scalar_alpha_policy"]["alpha_preserved_in_png"]
    assert not record["map_metadata"]["height"]["transforms"]
    assert original_hashes == {p.name: dataset.file_sha256(p) for p in folder.iterdir()}
    del record["map_metadata"]["height"]["scalar_alpha_policy"]
    with pytest.raises(ValueError, match="declared independent"):
        load_pair(sample, torch.device("cpu"))


def test_meaningful_scalar_alpha_requires_review_before_preparation(tmp_path):
    folder, _, source = material(tmp_path, alpha=65526)
    before = {p.name: dataset.file_sha256(p) for p in folder.iterdir()}
    with pytest.raises(ValueError, match="non-opaque alpha"):
        dataset.prepare_material(source, tmp_path / "dataset", {}, True, .2)
    assert before == {p.name: dataset.file_sha256(p) for p in folder.iterdir()}


@pytest.mark.parametrize("channels", (2, 4))
def test_explicit_scalar_transfer_keeps_auxiliary_alpha_untouched(channels):
    values = np.full((3, 3, channels), 32768, dtype=np.uint16)
    values[..., -1] = 65531
    result, transforms, _ = dataset.transformed_crop(values, "height", {"suffix": "disp"}, "srgb_to_linear")
    np.testing.assert_array_equal(result[..., -1], values[..., -1])
    assert np.all(result[..., 0] < values[..., 0]) and transforms


def test_decoder_rejects_nonreplicated_gray_expansion(tmp_path, monkeypatch):
    path = tmp_path / "original.png"
    independent_la_png(path, np.full((2, 3, 2), 65535, dtype=np.uint16))
    wrong = np.full((2, 3, 4), 65535, dtype=np.uint16)
    wrong[..., 1] -= 1
    monkeypatch.setattr(dataset.cv2, "imdecode", lambda *_args: wrong)
    with pytest.raises(ValueError, match=r"grayscale\+alpha values"):
        dataset.read_png(path)


def test_eight_bit_displacement_is_excluded_without_deleting_other_maps(tmp_path):
    folder, _, _ = material(tmp_path)
    height = folder / "surface_disp_4k.png"
    height.unlink()
    dataset.write_png(height, np.full((32, 32, 1), 127, dtype=np.uint8))
    before = {p.name: dataset.file_sha256(p) for p in folder.iterdir()}
    source = dataset.discover_materials(folder.parent, 16)[0]
    assert not source["ready"] and any("8-bit height is excluded" in p for p in source["problems"])
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="original 16-bit"):
        dataset.prepare_material(source, output, {}, True, .2)
    assert not output.exists()
    assert before == {p.name: dataset.file_sha256(p) for p in folder.iterdir()}
