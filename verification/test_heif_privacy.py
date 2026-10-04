"""Byte-level privacy regression, plus optional real Apple image integration."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from ipde.heif_privacy import (HeifPrivacyError, _boxes, _one, _orientation_exif,
                               _parse, _private_icc, privacy_heif_bytes, scrub_heif)


def box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def exif(orientation=1):
    return b"\0\0\0\x06Exif\0\0MM\0*\0\0\0\x08\0\x01" + struct.pack(">HHIHHI", 0x112, 3, 1, orientation, 0, 0) + b"private-date GPS private-UUID"


def xmp(functional=True):
    field = '<p:FloatMinValue>0.535156</p:FloatMinValue>' if functional else '<dc:creator>Private Person</dc:creator>'
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Private software">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="private-photo-UUID" '
            'xmlns:p="http://ns.apple.com/pixeldatainfo/1.0/" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">' + field +
            '<dc:description>Private Address</dc:description></rdf:Description></rdf:RDF></x:xmpmeta>').encode()


def fixture(*, orientation=1, functional=True, unknown_top=False, property_box=b""):
    image = b"fake compressed samples 01234567-89ab-cdef-0123-456789abcdef"
    payloads = [(1, b"hvc1", image), (2, b"Exif", exif(orientation)), (3, b"mime", xmp(functional))]
    entries = []
    for identifier, kind, payload in payloads:
        name = b"private-name" if identifier == 1 else b""
        info = b"\x02\0\0\0" + struct.pack(">HH4s", identifier, 0, kind) + name + b"\0"
        if kind == b"mime":
            info += b"application/rdf+xml\0"
        entries.append(box(b"infe", info))
    iinf = box(b"iinf", bytes(4) + struct.pack(">H", len(entries)) + b"".join(entries))
    hdlr = box(b"hdlr", bytes(8) + b"pict" + bytes(12) + b"Private handler name\0")
    pitm = box(b"pitm", bytes(4) + struct.pack(">H", 1))
    iprp = box(b"iprp", box(b"ipco", box(b"ispe", bytes(4) + struct.pack(">II", 2, 1)) + property_box) +
               box(b"ipma", bytes(4) + struct.pack(">I", 1) + struct.pack(">HB", 1, 1) + b"\x81"))
    iref = box(b"iref", bytes(4) + b"".join(box(b"cdsc", struct.pack(">HHH", identifier, 1, 1)) for identifier in (2, 3)))
    ftyp = box(b"ftyp", b"heic" + bytes(4) + b"mif1heic")
    padding = box(b"free", b"Private old padding")
    def meta(offset):
        rows = []
        for identifier, _, payload in payloads:
            rows.append(struct.pack(">HHHHII", identifier, 0, 0, 1, offset, len(payload)))
            offset += len(payload)
        iloc = box(b"iloc", b"\x01\0\0\0\x44\0" + struct.pack(">H", len(rows)) + b"".join(rows))
        return box(b"meta", bytes(4) + hdlr + pitm + iinf + iref + iprp + padding + iloc)
    offset = len(ftyp) + len(meta(0)) + 8
    source = ftyp + meta(offset) + box(b"mdat", b"".join(payload for _, _, payload in payloads)) + box(b"free", b"Private tail padding")
    if unknown_top:
        source += box(b"abcd", b"Unknown data")
    return source


def icc_profile(extra_tag=None):
    description = b"mluc" + bytes(4) + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", 28, 28) + "Private Person".encode("utf-16-be")
    transform = b"XYZ " + bytes(4) + struct.pack(">iii", 63190, 65536, 54060)
    curve = b"curv" + bytes(4) + struct.pack(">IH", 1, 256)
    payloads = [(b"desc", description), (b"cprt", description), (b"wtpt", transform),
                (b"kTRC", curve), (b"dmnd", b"text" + bytes(4) + b"Private Device\0")]
    if extra_tag:
        payloads.append(extra_tag)
    header = bytearray(128); header[4:8] = b"USER"; header[36:40] = b"acsp"
    header[24:36] = struct.pack(">6H", 2026, 10, 4, 12, 13, 14)
    header[48:56] = b"Private!"; header[80:100] = b"Private profile UUID"
    offset = 132 + 12 * len(payloads); records = []; bodies = []
    for kind, payload in payloads:
        records.append(kind + struct.pack(">II", offset, len(payload)))
        bodies.append(payload); offset += len(payload)
    profile = bytes(header) + len(payloads).to_bytes(4, "big") + b"".join(records) + b"".join(bodies)
    return len(profile).to_bytes(4, "big") + profile[4:]


class HeifPrivacyTests(unittest.TestCase):
    def test_icc_identity_erased_and_transform_bytes_preserved(self):
        source = icc_profile(); result = _private_icc(source)
        self.assertEqual(len(source), len(result))
        self.assertNotIn("Private Person".encode("utf-16-be"), result)
        self.assertNotIn(b"Private", result)
        self.assertNotIn(b"USER", result)
        self.assertEqual(result[24:36], struct.pack(">6H", 1970, 1, 1, 0, 0, 0))
        self.assertEqual(result[48:56], bytes(8))
        self.assertEqual(result[80:128], bytes(48))
        for position in range(132, 132 + 12 * int.from_bytes(source[128:132], "big"), 12):
            if source[position:position + 4] in {b"wtpt", b"kTRC"}:
                offset, length = struct.unpack_from(">II", source, position + 4)
                self.assertEqual(source[offset:offset + length], result[offset:offset + length])
        _, report = privacy_heif_bytes(fixture(property_box=box(b"colr", b"prof" + source)))
        self.assertEqual(report["scrubbed_icc_profiles"], 1)

    def test_unknown_icc_tag_and_text_in_color_transform_refuse(self):
        for tag in ((b"usr1", b"text" + bytes(4) + b"Private Person\0"),
                    (b"rXYZ", b"text" + bytes(4) + b"Private Person\0")):
            with self.subTest(tag=tag[0]), self.assertRaises(HeifPrivacyError):
                _private_icc(icc_profile(tag))

    def test_auxiliary_known_types_and_numeric_depth_sei_only(self):
        depth_subtype = bytes.fromhex("000000110000000d4e01b109351e78900103ec5020")
        for kind, subtype in ((b"urn:com:apple:photo:2020:aux:hdrgainmap", b""),
                              (b"urn:mpeg:hevc:2015:auxid:2", depth_subtype)):
            with self.subTest(kind=kind):
                prop = box(b"auxC", bytes(4) + kind + b"\0" + subtype)
                output, _ = privacy_heif_bytes(fixture(property_box=prop))
                self.assertIn(prop, output)
        for kind, subtype in ((b"urn:user:Private Person", b""),
                              (b"urn:com:apple:photo:2020:aux:hdrgainmap", b"Private Person"),
                              (b"urn:mpeg:hevc:2015:auxid:2", depth_subtype[:10] + b"\x05" + depth_subtype[11:]),
                              (b"urn:mpeg:hevc:2015:auxid:2", depth_subtype[:-1])):
            with self.subTest(kind=kind, subtype=subtype), self.assertRaises(HeifPrivacyError):
                privacy_heif_bytes(fixture(property_box=box(b"auxC", bytes(4) + kind + b"\0" + subtype)))

    def test_removal_erases_payloads_records_names_and_padding_without_moving_samples(self):
        source = fixture()
        output, report = privacy_heif_bytes(source)
        self.assertEqual(len(source), len(output))
        self.assertNotIn(b"Private", output)
        self.assertNotIn(b"private", output)
        _, _, items, locations = _parse(output)
        self.assertEqual(set(items), {1, 3})
        old_location = _parse(source)[3][0]
        new_location = locations[0]
        self.assertEqual(old_location.ranges, new_location.ranges)
        for start, end in old_location.ranges:
            self.assertEqual(source[start:end], output[start:end])
        self.assertIn(b"01234567-89ab-cdef-0123-456789abcdef", output)
        self.assertTrue(report["compressed_payload_bit_exact"])
        self.assertEqual(report["randomized_item_names"], 1)
        self.assertIn(b"0.535156", output)

    def test_nonfunctional_xmp_is_physically_removed(self):
        output, report = privacy_heif_bytes(fixture(functional=False))
        self.assertEqual(set(_parse(output)[2]), {1})
        self.assertEqual(report["removed_metadata_item_count"], 2)
        self.assertNotIn(b"Private", output)

    def test_orientation_only_exif_keeps_all_nondefault_transform_values(self):
        for orientation in range(2, 9):
            with self.subTest(orientation=orientation):
                output, report = privacy_heif_bytes(fixture(orientation=orientation))
                _, _, items, locations = _parse(output)
                self.assertEqual(items[2].kind, b"Exif")
                location = next(location for location in locations if location.identifier == 2)
                payload = b"".join(output[start:end] for start, end in location.ranges)
                self.assertEqual(int.from_bytes(payload[28:30], "big"), orientation)
                self.assertNotIn(b"private-date", output)
                self.assertEqual(report["orientation_only_exif_items"], 1)

    def test_unknown_and_truncated_structures_refuse_instead_of_claiming_privacy(self):
        for data in (fixture(unknown_top=True), fixture()[:-3], b"not HEIF"):
            with self.subTest(length=len(data)), self.assertRaises(HeifPrivacyError):
                privacy_heif_bytes(data)

    def test_overlapping_metadata_and_image_extents_refuse(self):
        source = bytearray(fixture())
        _, tables, _, locations = _parse(bytes(source))
        iloc = tables[b"iloc"]
        # v1 header8; each row16; offset field is eight bytes into the row.
        image_offset = locations[0].ranges[0][0]
        struct.pack_into(">I", source, iloc.payload + 8 + 16 + 8, image_offset)
        with self.assertRaises(HeifPrivacyError):
            privacy_heif_bytes(bytes(source))

    def test_existing_destination_and_source_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.heic"; source.write_bytes(fixture())
            destination = Path(temporary) / "output.heic"; destination.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                scrub_heif(source, destination)
            with self.assertRaises(HeifPrivacyError):
                scrub_heif(source, source)
            self.assertEqual(source.read_bytes(), fixture())
            self.assertEqual(destination.read_bytes(), b"existing")


class RealApplePrivacyTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "Apple ImageIO integration requires macOS")
    def test_real_heic_exif6_and_container_rotation_keep_effective_orientation(self):
        import numpy as np
        import pillow_heif
        from PIL import Image
        from ipde.photo_workflow import discover_photo, run_native, _orientation_inventory
        from ipde.formats import arrays_bit_equal
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for native_rotation in (False, True):
                with self.subTest(native_rotation=native_rotation):
                    source = root / f"source-{native_rotation}.heic"
                    metadata = Image.Exif(); metadata[274] = 6 if native_rotation else 1
                    metadata[306] = "2026:10:04 12:13:14"
                    image = pillow_heif.from_bytes("RGB", (16, 24), np.arange(16 * 24 * 3, dtype=np.uint8).tobytes())
                    image.save(source, exif=metadata.tobytes(), quality=-1)
                    data = bytearray(source.read_bytes())
                    _, tables, items, locations = _parse(bytes(data))
                    ipco = _one(_boxes(data, tables[b"iprp"].payload, tables[b"iprp"].end), b"ipco")
                    self.assertEqual(any(prop.kind == b"irot" for prop in _boxes(data, ipco.payload, ipco.end)), native_rotation)
                    location = next(value for value in locations if items[value.identifier].kind == b"Exif")
                    self.assertEqual(len(location.ranges), 1)
                    start, end = location.ranges[0]
                    tiff = start + 4 + int.from_bytes(data[start:start + 4], "big")
                    endian = "little" if data[tiff:tiff + 2] == b"II" else "big"
                    directory = tiff + int.from_bytes(data[tiff + 4:tiff + 8], endian)
                    count = int.from_bytes(data[directory:directory + 2], endian)
                    for position in range(directory + 2, directory + 2 + count * 12, 12):
                        if int.from_bytes(data[position:position + 2], endian) == 274:
                            data[position + 8:position + 10] = (6).to_bytes(2, endian)
                    source.write_bytes(data)
                    destination = root / f"private-{native_rotation}.heic"
                    report = scrub_heif(source, destination)
                    self.assertEqual(report["orientation_only_exif_items"], 1)
                    after = destination.read_bytes(); after_location = next(value for value in _parse(after)[3] if value.identifier == location.identifier)
                    self.assertEqual(_orientation_exif(bytes(data[start:end])),
                                     _orientation_exif(b"".join(after[a:b] for a, b in after_location.ranges)))
                    before_inventory = run_native("inventory", source, root / "before")
                    after_inventory = run_native("inventory", destination, root / "after")
                    self.assertEqual(_orientation_inventory(before_inventory), _orientation_inventory(after_inventory))
                    self.assertNotIn(b"2026:10:04", after)
                    self.assertTrue(arrays_bit_equal(discover_photo(source).assets[0].array,
                                                     discover_photo(destination).assets[0].array))

    def test_portrait_and_spatial_inventory_orientation_calibration_and_samples(self):
        from ipde.photo_workflow import discover_photo, run_native
        from ipde.formats import arrays_bit_equal
        fixtures = [Path("/opt/ipde/IMG_6678.HEIC"),
                    Path("/Users/andrewsmith/Downloads/IMG_0835.HEIC"),
                    Path("/Users/andrewsmith/Downloads/IMG_1515.HEIC")]
        fixtures = [path for path in fixtures if path.is_file()]
        if not fixtures:
            self.skipTest("Local real Apple fixtures are unavailable")
        for source in fixtures:
            with self.subTest(source=source.name), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "random.heic"
                original = source.read_bytes()
                report = scrub_heif(source, destination)
                old = discover_photo(source); new = discover_photo(destination)
                self.assertEqual(len(old.top_level_images), len(new.top_level_images))
                self.assertEqual(old.primary_index, new.primary_index)
                self.assertEqual(old.spatial_photo, new.spatial_photo)
                self.assertEqual(len(old.assets), len(new.assets))
                for before, after in zip(old.assets, new.assets, strict=True):
                    self.assertEqual((before.kind, before.semantic_name, before.parent_image_index, before.source_bit_depth),
                                     (after.kind, after.semantic_name, after.parent_image_index, after.source_bit_depth))
                    self.assertTrue(arrays_bit_equal(before.array, after.array), before.semantic_name)
                props = run_native("inventory", destination, Path(temporary) / "inventory.json")["properties"]
                source_props = run_native("inventory", source, Path(temporary) / "source-inventory.json")["properties"]
                self.assertEqual([image.get("Orientation", 1) for image in source_props["images"]],
                                 [image.get("Orientation", 1) for image in props["images"]])
                def check(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            self.assertNotIn(key, {"{Exif}", "{GPS}", "{IPTC}", "{MakerApple}", "Make", "Model", "DateTime", "DateTimeOriginal"})
                            check(child)
                    elif isinstance(value, list):
                        for child in value:
                            check(child)
                check(props)
                self.assertTrue(report["privacy_verified"])
                self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
