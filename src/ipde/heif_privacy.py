"""Remove descriptive HEIF metadata without decoding/re-encoding image samples.

Supported still-image BMFF layouts are parsed before mutation. Tables shrink
inside their original spans using sibling ``free`` boxes; media offsets and all
retained image-item bytes remain unchanged. Unsupported structures are refused.
This preserves functional color/calibration/auxiliary descriptions, not anonymity
of visible people or mathematical camera fingerprints.

Box layouts follow libheif's primary implementation:
https://github.com/strukturag/libheif/blob/master/libheif/box.cc
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import secrets
import struct
from typing import Any
import xml.etree.ElementTree as ET


class HeifPrivacyError(RuntimeError):
    """The container cannot be scrubbed while proving payload preservation."""


@dataclass(frozen=True)
class _Box:
    start: int
    end: int
    kind: bytes
    header: int

    @property
    def payload(self) -> int:
        return self.start + self.header


class _Cursor:
    def __init__(self, data: bytes, start: int, end: int):
        self.data, self.pos, self.end = data, start, end

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > self.end:
            raise HeifPrivacyError("Truncated HEIF table")
        value = self.data[self.pos:self.pos + size]; self.pos += size
        return value

    def number(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big")

    def cstring(self) -> tuple[bytes, tuple[int, int]]:
        end = self.data.find(b"\0", self.pos, self.end)
        if end < 0:
            raise HeifPrivacyError("Unterminated HEIF item string")
        span = (self.pos, end)
        value = self.take(end - self.pos); self.take(1)
        return value, span

    def finish(self) -> None:
        if self.pos != self.end:
            raise HeifPrivacyError("Unexpected trailing HEIF table fields")


def _boxes(data: bytes, start: int, end: int) -> list[_Box]:
    result = []
    cursor = _Cursor(data, start, end)
    while cursor.pos < end:
        offset = cursor.pos
        size = cursor.number(4); kind = cursor.take(4); header = 8
        if size == 1:
            size = cursor.number(8); header = 16
        if size == 0:
            size = end - offset
        if kind == b"uuid":
            cursor.take(16); header += 16
        if size < header or size > end - offset:
            raise HeifPrivacyError("Invalid HEIF box size")
        result.append(_Box(offset, offset + size, kind, header))
        cursor.pos = offset + size
    return result


def _one(boxes: list[_Box], kind: bytes, *, optional: bool = False) -> _Box | None:
    matches = [box for box in boxes if box.kind == kind]
    if len(matches) != 1:
        if optional and not matches:
            return None
        raise HeifPrivacyError(f"Expected one {kind.decode('ascii')} box")
    return matches[0]


def _full(data: bytes, box: _Box, versions: set[int], allowed_flags: int = 0) -> _Cursor:
    cursor = _Cursor(data, box.payload, box.end)
    version = cursor.number(1); flags = cursor.number(3)
    if version not in versions or flags & ~allowed_flags:
        raise HeifPrivacyError(f"Unsupported {box.kind.decode('ascii')} version or flags")
    return cursor


@dataclass
class _Item:
    identifier: int
    kind: bytes
    box: _Box
    name_span: tuple[int, int]
    content_type: bytes = b""
    uri: bytes = b""


@dataclass
class _Location:
    identifier: int
    raw: bytes
    ranges: list[tuple[int, int]]


_IMAGES = {b"hvc1", b"av01", b"grid", b"iden", b"iovl", b"tmap"}
_PROPERTIES = {b"colr", b"ispe", b"irot", b"imir", b"pixi", b"auxC", b"hvcC", b"av1C",
               b"clap", b"pasp", b"clli", b"mdcv"}
# These UUIDs identify box schemas, not individual photographs. Replacing them
# would destroy camera calibration and Apple stereo-group interpretation.
_SPATIAL_PROPERTIES = {
    bytes.fromhex("22cc04c7d6d94e079d904eb6ecbaf3a3"): {16},  # cmin
    bytes.fromhex("4363e9145b7d4aab97aebea69803b434"): {4, 8},  # cmex
    bytes.fromhex("de22508536cb436587432f8705e7c78a"): {4},
    bytes.fromhex("a33e221ad1d0464db252780190d1734d"): {7},
}
_SPATIAL_GROUP = bytes.fromhex("524a52f437b347d09637933d05ac9b68")
_AUXILIARY_TYPES = {
    b"urn:mpeg:hevc:2015:auxid:1", b"urn:mpeg:hevc:2015:auxid:2",
    b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha",
    b"urn:com:apple:photo:2018:aux:portraiteffectsmatte",
    b"urn:com:apple:photo:2020:aux:hdrgainmap",
    b"tag:apple.com,2023:photo:aux:linearthumbnail",
    b"tag:apple.com,2023:photo:aux:styledeltamap",
} | {b"urn:com:apple:photo:" + year + b":aux:semantic" + semantic + b"matte"
     for year in (b"2019", b"2020") for semantic in (b"hair", b"skin", b"teeth", b"glasses", b"sky")}


def _validate_auxiliary(data: bytes, box: _Box) -> None:
    cursor = _full(data, box, {0})
    kind, _ = cursor.cstring()
    if kind not in _AUXILIARY_TYPES:
        raise HeifPrivacyError("Unknown auxiliary URI could contain identifying metadata")
    if cursor.pos == cursor.end:
        return
    if kind != b"urn:mpeg:hevc:2015:auxid:2":
        raise HeifPrivacyError("Unknown auxiliary subtype could contain identifying metadata")
    # HEIF's HEVC depth subtype contains size-delimited SEI NAL units. Only
    # depth_representation_info (HEVC payload type 177) is functional here;
    # user-data, UUID, or other SEI messages are refused. Samples are untouched.
    while cursor.pos < cursor.end:
        length = cursor.number(4)
        if length < 7:
            raise HeifPrivacyError("Malformed depth auxiliary subtype")
        segment = cursor.take(length)
        units = _Cursor(segment, 0, len(segment))
        while units.pos < units.end:
            nal = units.take(units.number(4))
            if len(nal) < 4 or nal[:2] != b"\x4e\x01":
                raise HeifPrivacyError("Unsupported depth auxiliary SEI header")
            rbsp = nal[2:].replace(b"\0\0\x03", b"\0\0")
            messages = _Cursor(rbsp, 0, len(rbsp))
            while messages.pos < messages.end:
                if rbsp[messages.pos:] == b"\x80":
                    messages.take(1); break
                value = messages.number(1); payload_type = value
                while value == 255:
                    value = messages.number(1); payload_type += value
                value = messages.number(1); payload_size = value
                while value == 255:
                    value = messages.number(1); payload_size += value
                if payload_type != 177 or not 1 <= payload_size <= 64:
                    raise HeifPrivacyError("Depth auxiliary subtype contains unsupported SEI metadata")
                messages.take(payload_size)


def _group_ids(data: bytes, box: _Box) -> set[int]:
    identifiers = set()
    for group in _boxes(data, box.payload, box.end):
        if group.kind not in {b"altr", b"ster", b"uuid"}:
            raise HeifPrivacyError("Unknown HEIF image grouping")
        if group.kind == b"uuid" and data[group.payload - 16:group.payload] != _SPATIAL_GROUP:
            raise HeifPrivacyError("Unknown UUID group could contain identifying metadata")
        cursor = _full(data, group, {0}, 2 if group.kind == b"uuid" else 0)
        identifier = cursor.number(4)
        if identifier in identifiers:
            raise HeifPrivacyError("Duplicate HEIF group identifier")
        identifiers.add(identifier)
        cursor.take(cursor.number(4) * 4); cursor.finish()
    return identifiers


def _validate_properties(data: bytes, box: _Box, item_ids: set[int]) -> None:
    children = _boxes(data, box.payload, box.end)
    if any(child.kind not in {b"ipco", b"ipma", b"free", b"skip"} for child in children):
        raise HeifPrivacyError("Unknown HEIF property container")
    properties = _one(children, b"ipco")
    for prop in _boxes(data, properties.payload, properties.end):
        if prop.kind == b"uuid":
            schema = data[prop.payload - 16:prop.payload]
            if prop.end - prop.payload not in _SPATIAL_PROPERTIES.get(schema, set()):
                raise HeifPrivacyError("Unknown UUID property could contain identifying metadata")
        elif prop.kind not in _PROPERTIES:
            raise HeifPrivacyError(f"Unsupported image property {prop.kind!r}")
        elif prop.kind == b"auxC":
            _validate_auxiliary(data, prop)
        elif prop.kind == b"colr":
            color = data[prop.payload:prop.payload + 4]
            if color not in {b"prof", b"rICC", b"nclx"} or (color == b"nclx" and prop.end - prop.payload != 11):
                raise HeifPrivacyError("Unsupported or extended color metadata")
        elif prop.kind in {b"irot", b"imir"}:
            payload = data[prop.payload:prop.end]
            if len(payload) != 1 or payload[0] > (3 if prop.kind == b"irot" else 1):
                raise HeifPrivacyError("Unsupported image transform metadata")
        elif prop.kind == b"ispe":
            cursor = _full(data, prop, {0})
            if not cursor.number(4) or not cursor.number(4):
                raise HeifPrivacyError("Invalid HEIF image dimensions")
            cursor.finish()
        elif prop.kind == b"pixi":
            cursor = _full(data, prop, {0})
            count = cursor.number(1)
            if not 1 <= count <= 16 or any(not 1 <= cursor.number(1) <= 64 for _ in range(count)):
                raise HeifPrivacyError("Unsupported HEIF pixel channels")
            cursor.finish()
        elif prop.kind in {b"clap", b"pasp", b"clli", b"mdcv"}:
            if prop.end - prop.payload != {b"clap": 32, b"pasp": 8, b"clli": 4, b"mdcv": 24}[prop.kind]:
                raise HeifPrivacyError("Unsupported extended image property")
    associations = _one(children, b"ipma")
    version = data[associations.payload]
    cursor = _full(data, associations, {0, 1}, 1)
    flags = int.from_bytes(data[associations.payload + 1:associations.payload + 4], "big")
    count = cursor.number(4)
    for _ in range(count):
        identifier = cursor.number(2 if version == 0 else 4)
        if identifier not in item_ids:
            raise HeifPrivacyError("HEIF property association references a missing item")
        for _ in range(cursor.number(1)):
            cursor.number(2 if flags & 1 else 1)
    cursor.finish()


def _parse(data: bytes) -> tuple[list[_Box], dict[bytes, _Box], dict[int, _Item], list[_Location]]:
    top = _boxes(data, 0, len(data))
    if any(box.kind not in {b"ftyp", b"meta", b"mdat", b"free", b"skip"} for box in top):
        raise HeifPrivacyError("Unsupported top-level HEIF structure; privacy output refused")
    _one(top, b"ftyp")
    meta = _one(top, b"meta")
    _full(data, meta, {0})
    children = _boxes(data, meta.payload + 4, meta.end)
    allowed = {b"hdlr", b"dinf", b"pitm", b"iinf", b"iref", b"iprp", b"grpl", b"idat", b"iloc", b"free", b"skip"}
    if any(box.kind not in allowed for box in children):
        raise HeifPrivacyError("Unknown HEIF metadata structure; privacy output refused")
    tables = {kind: _one(children, kind) for kind in (b"hdlr", b"pitm", b"iinf", b"iloc", b"iprp")}
    for kind in (b"iref", b"grpl", b"dinf", b"idat"):
        value = _one(children, kind, optional=True)
        if value is not None:
            tables[kind] = value
    hdlr = _full(data, tables[b"hdlr"], {0})
    if hdlr.number(4) or hdlr.take(4) != b"pict" or any(hdlr.take(12)):
        raise HeifPrivacyError("Only ordinary unencrypted still-image HEIF handlers are supported")
    if b"dinf" in tables:
        data_references = _boxes(data, tables[b"dinf"].payload, tables[b"dinf"].end)
        if any(box.kind != b"dref" for box in data_references):
            raise HeifPrivacyError("Unknown HEIF data-reference structure")
        dref = _one(data_references, b"dref")
        cursor = _full(data, dref, {0}); count = cursor.number(4)
        refs = _boxes(data, cursor.pos, cursor.end)
        if len(refs) != count:
            raise HeifPrivacyError("Malformed HEIF data-reference table")
        for ref in refs:
            if ref.kind != b"url " or data[ref.payload:ref.end] != b"\0\0\0\1":
                raise HeifPrivacyError("External HEIF data references are unsupported")
    iinf = tables[b"iinf"]; version = data[iinf.payload]
    cursor = _full(data, iinf, {0, 1}); count = cursor.number(2 if version == 0 else 4)
    entries = _boxes(data, cursor.pos, cursor.end)
    if len(entries) != count:
        raise HeifPrivacyError("Malformed HEIF item inventory")
    items = {}
    for entry in entries:
        if entry.kind != b"infe":
            raise HeifPrivacyError("Unsupported HEIF item inventory record")
        version = data[entry.payload]; cursor = _full(data, entry, {2, 3}, 1)
        identifier = cursor.number(2 if version == 2 else 4)
        if identifier in items or cursor.number(2):
            raise HeifPrivacyError("Duplicate or encrypted HEIF item")
        kind = cursor.take(4)
        _, name = cursor.cstring()
        item = _Item(identifier, kind, entry, name)
        if kind == b"mime":
            item.content_type, _ = cursor.cstring()
            if cursor.pos < cursor.end:
                encoding, _ = cursor.cstring()
                if encoding:
                    raise HeifPrivacyError("Compressed descriptive metadata is unsupported")
        elif kind == b"uri ":
            item.uri, _ = cursor.cstring()
        elif kind not in _IMAGES | {b"Exif"}:
            raise HeifPrivacyError(f"Unsupported HEIF item type {kind!r}")
        cursor.finish(); items[identifier] = item
    pitm = tables[b"pitm"]; version = data[pitm.payload]
    cursor = _full(data, pitm, {0, 1}); primary = cursor.number(2 if version == 0 else 4); cursor.finish()
    if primary not in items or items[primary].kind not in _IMAGES:
        raise HeifPrivacyError("Invalid primary HEIF image")
    # HEIF entity groups can have property associations of their own (Apple's
    # spatial group carries stereo interpretation flags through ipma).
    group_ids = _group_ids(data, tables[b"grpl"]) if b"grpl" in tables else set()
    _validate_properties(data, tables[b"iprp"], set(items) | group_ids)
    iloc = tables[b"iloc"]; version = data[iloc.payload]
    cursor = _full(data, iloc, {0, 1, 2}); sizes = cursor.number(2)
    offset_size, length_size, base_size = sizes >> 12, (sizes >> 8) & 15, (sizes >> 4) & 15
    index_size = (sizes & 15) if version else 0
    if any(size not in {0, 4, 8} for size in (offset_size, length_size, base_size, index_size)) or not length_size:
        raise HeifPrivacyError("Unsupported HEIF extent widths")
    count = cursor.number(2 if version < 2 else 4)
    locations = []; seen = set()
    media = [box for box in top if box.kind == b"mdat"]
    for _ in range(count):
        start = cursor.pos; identifier = cursor.number(2 if version < 2 else 4)
        if identifier not in items or identifier in seen:
            raise HeifPrivacyError("HEIF location references a missing or duplicate item")
        seen.add(identifier)
        method = cursor.number(2) if version else 0
        if method not in {0, 1} or cursor.number(2):
            raise HeifPrivacyError("External, reserved or indirect HEIF extents are unsupported")
        base = cursor.number(base_size); ranges = []
        for _ in range(cursor.number(2)):
            if cursor.number(index_size):
                raise HeifPrivacyError("Indexed HEIF extents are unsupported")
            offset = cursor.number(offset_size); length = cursor.number(length_size)
            if method == 1:
                if b"idat" not in tables:
                    raise HeifPrivacyError("HEIF extent references missing idat")
                containers = [tables[b"idat"]]; offset += containers[0].payload
            else:
                containers = media
            offset += base
            if not length or not any(container.payload <= offset and offset + length <= container.end for container in containers):
                raise HeifPrivacyError("HEIF extent is empty or escapes its media payload")
            ranges.append((offset, offset + length))
        if not ranges:
            raise HeifPrivacyError("HEIF item has no located payload")
        locations.append(_Location(identifier, data[start:cursor.pos], ranges))
    cursor.finish()
    if set(items) != seen:
        raise HeifPrivacyError("HEIF inventory and located payloads differ")
    return top, tables, items, locations


_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XMP = "adobe:ns:meta/"
_FUNCTIONAL = {
    "http://ns.apple.com/pixeldatainfo/1.0/": {"IntMaxValue", "StoredFormat", "NativeFormat", "IntMinValue", "FloatMaxValue", "FloatMinValue", "AuxiliaryImageType", "AuxiliaryImageSubType"},
    "http://ns.apple.com/depthData/1.0/": {"IntrinsicMatrixReferenceWidth", "DepthDataVersion", "Quality", "IntrinsicMatrix", "IntrinsicMatrixReferenceHeight", "InverseLensDistortionCoefficients", "LensDistortionCenterOffsetX", "LensDistortionCoefficients", "Accuracy", "PixelSize", "Filtered", "ExtrinsicMatrix", "LensDistortionCenterOffsetY"},
    "http://ns.apple.com/depthBlurEffect/1.0/": {"RenderingParameters", "SimulatedAperture"},
    "http://ns.apple.com/portraitLightingEffect/2.0/": {"EffectStrength"},
    "http://ns.apple.com/portraitEffectsMatte/1.0/": {"PortraitEffectsMatteVersion"},
    "http://ns.apple.com/semanticSegmentationMatte/1.0/": {"SemanticSegmentationMatteVersion"},
    "http://ns.apple.com/HDRGainMap/1.0/": {"HDRGainMapVersion", "HDRGainMapHeadroom"},
    "http://ns.adobe.com/hdr-gain-map/1.0/": {"Version", "BaseRenditionIsHDR", "HDRCapacityMin", "HDRCapacityMax", "GainMapMin", "GainMapMax", "Gamma", "OffsetSDR", "OffsetHDR"},
}
_TEXT_VALUES = {"depth", "disparity", "portraiteffectsmatte", "high", "low", "relative", "absolute", "True", "False", "true", "false"}


def _qualified(tag: str) -> tuple[str, str]:
    if tag.startswith("{") and "}" in tag:
        namespace, name = tag[1:].split("}", 1)
        return namespace, name
    return "", tag


def _functional_xmp(payload: bytes) -> bytes | None:
    if b"<!DOCTYPE" in payload or b"<!ENTITY" in payload:
        raise HeifPrivacyError("Unsupported XMP document declarations")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise HeifPrivacyError("Cannot inspect HEIF XMP metadata") from exc
    if root.tag not in {f"{{{_XMP}}}xmpmeta", f"{{{_RDF}}}RDF"}:
        raise HeifPrivacyError("Unsupported XMP root structure")
    ET.register_namespace("x", _XMP); ET.register_namespace("rdf", _RDF)
    for index, namespace in enumerate(_FUNCTIONAL):
        ET.register_namespace(f"f{index}", namespace)
    new_root = ET.Element(f"{{{_XMP}}}xmpmeta")
    rdf = ET.SubElement(new_root, f"{{{_RDF}}}RDF")
    description = ET.SubElement(rdf, f"{{{_RDF}}}Description", {f"{{{_RDF}}}about": ""})
    for old in root.iter(f"{{{_RDF}}}Description"):
        fields = list(old)
        for tag, value in old.attrib.items():
            namespace, name = _qualified(tag)
            if namespace in _FUNCTIONAL:
                element = ET.Element(tag); element.text = value; fields.append(element)
        for field in fields:
            namespace, name = _qualified(field.tag)
            if namespace not in _FUNCTIONAL:
                if namespace.startswith("http://ns.apple.com/"):
                    raise HeifPrivacyError(f"Unknown Apple XMP namespace {namespace}; functional metadata may be required")
                continue
            if name not in _FUNCTIONAL[namespace]:
                raise HeifPrivacyError(f"Unknown functional XMP field {name}")
            # Retain exact numerical text and RDF sequences. Arbitrary strings,
            # links, identifiers, and hidden attributes are not accepted here.
            kept = copy.deepcopy(field)
            for element in kept.iter():
                ns, local = _qualified(element.tag)
                if element is not kept and (ns != _RDF or local not in {"Seq", "Bag", "li"}):
                    raise HeifPrivacyError("Unknown nested functional XMP structure")
                if element.attrib:
                    raise HeifPrivacyError("Unknown functional XMP attributes")
                text = (element.text or "").strip()
                if text:
                    if local == "RenderingParameters":
                        if not re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
                            raise HeifPrivacyError("Invalid portrait rendering parameters")
                    elif text not in _TEXT_VALUES:
                        try:
                            if not math.isfinite(float(text)):
                                raise ValueError
                        except ValueError as exc:
                            raise HeifPrivacyError(f"Unexpected value in functional XMP field {local}") from exc
                element.tail = None
            description.append(kept)
    return ET.tostring(new_root, encoding="utf-8") if len(description) else None


def _replace_table(output: bytearray, box: _Box, payload: bytes) -> None:
    replacement = struct.pack(">I4s", len(payload) + 8, box.kind) + payload
    spare = box.end - box.start - len(replacement)
    if spare < 0 or 0 < spare < 8:
        raise HeifPrivacyError("Scrubbed HEIF table cannot fit without moving image offsets")
    if spare:
        replacement += struct.pack(">I4s", spare, b"free") + bytes(spare - 8)
    output[box.start:box.end] = replacement


def _references(data: bytes, box: _Box, removed: set[int], identifiers: set[int]) -> bytes:
    version = data[box.payload]; cursor = _full(data, box, {0, 1})
    entries = []
    for ref in _boxes(data, cursor.pos, cursor.end):
        if ref.kind not in {b"cdsc", b"auxl", b"dimg", b"thmb"}:
            raise HeifPrivacyError("Unknown HEIF item-reference meaning")
        part = _Cursor(data, ref.payload, ref.end)
        source = part.number(2 if version == 0 else 4)
        targets = [part.number(2 if version == 0 else 4) for _ in range(part.number(2))]
        part.finish()
        if source not in identifiers or any(target not in identifiers for target in targets):
            raise HeifPrivacyError("Dangling HEIF item reference")
        if source in removed or any(target in removed for target in targets):
            if ref.kind != b"cdsc" or source not in removed or any(target in removed for target in targets):
                raise HeifPrivacyError("Descriptive metadata is used as a functional HEIF image dependency")
            continue
        entries.append(data[ref.start:ref.end])
    return data[box.payload:box.payload + 4] + b"".join(entries)


def _orientation_exif(payload: bytes) -> bytes | None:
    """Retain only a structural TIFF orientation when it is not the default."""
    if len(payload) < 12:
        raise HeifPrivacyError("Truncated HEIF Exif payload")
    start = 4 + int.from_bytes(payload[:4], "big")
    if start + 8 > len(payload) or payload[start:start + 2] not in {b"II", b"MM"}:
        raise HeifPrivacyError("Unsupported HEIF Exif/TIFF header")
    endian = "little" if payload[start:start + 2] == b"II" else "big"
    if int.from_bytes(payload[start + 2:start + 4], endian) != 42:
        raise HeifPrivacyError("Unsupported HEIF Exif TIFF version")
    position = start + int.from_bytes(payload[start + 4:start + 8], endian)
    if position < start + 8 or position + 2 > len(payload):
        raise HeifPrivacyError("Invalid Exif orientation directory")
    count = int.from_bytes(payload[position:position + 2], endian); position += 2
    if position + count * 12 + 4 > len(payload):
        raise HeifPrivacyError("Truncated Exif orientation directory")
    orientation = None
    for offset in range(position, position + count * 12, 12):
        if int.from_bytes(payload[offset:offset + 2], endian) != 0x112:
            continue
        if orientation is not None or int.from_bytes(payload[offset + 2:offset + 4], endian) != 3 or int.from_bytes(payload[offset + 4:offset + 8], endian) != 1:
            raise HeifPrivacyError("Unsupported or duplicate Exif orientation")
        orientation = int.from_bytes(payload[offset + 8:offset + 10], endian)
        if orientation not in range(1, 9):
            raise HeifPrivacyError("Invalid Exif orientation")
    if orientation in {None, 1}:
        return None
    # HEIF Exif offset is measured from the byte following its four-byte offset.
    tiff = b"MM\0*\0\0\0\x08\0\x01" + struct.pack(">HHIHHI", 0x112, 3, 1, orientation, 0, 0)
    return b"\0\0\0\x06Exif\0\0" + tiff


def _write_payload(output: bytearray, ranges: list[tuple[int, int]], payload: bytes, pad: bytes) -> None:
    total = sum(end - start for start, end in ranges)
    if len(payload) > total:
        raise HeifPrivacyError("Functional metadata cannot fit its original payload")
    padded = payload + pad * (total - len(payload)); position = 0
    for start, end in ranges:
        output[start:end] = padded[position:position + end - start]; position += end - start


_ICC_FUNCTIONAL = {b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kTRC", b"rTRC", b"gTRC", b"bTRC",
                   b"chad", b"A2B0", b"A2B1", b"A2B2", b"B2A0", b"B2A1", b"B2A2", b"gamt",
                   b"cicp", b"HAGC", b"hdgm"}
_ICC_DESCRIPTIVE = {b"desc", b"cprt", b"dmnd", b"dmdd", b"vued", b"meta", b"pseq", b"psid"}
_ICC_TYPES = {
    **{kind: {b"XYZ "} for kind in (b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ")},
    **{kind: {b"curv", b"para"} for kind in (b"kTRC", b"rTRC", b"gTRC", b"bTRC")},
    **{kind: {b"mft1", b"mft2", b"mAB "} for kind in (b"A2B0", b"A2B1", b"A2B2")},
    **{kind: {b"mft1", b"mft2", b"mBA "} for kind in (b"B2A0", b"B2A1", b"B2A2")},
    b"chad": {b"sf32"}, b"gamt": {b"mft1", b"mft2", b"mAB "},
    b"cicp": {b"cicp"}, b"HAGC": {b"hagc"}, b"hdgm": {b"gmap"},
}


def _private_icc(profile: bytes) -> bytes:
    """Clear profile descriptions/identity while preserving every transform byte.

    Header/table layouts: https://www.color.org/specification/ICC.1-2022-05.pdf
    ICC permits zero signatures for CMM/manufacturer/model/creator and a zero ID.
    A fixed valid profile creation date replaces the original recorded time.
    """
    if len(profile) < 132 or int.from_bytes(profile[:4], "big") != len(profile) or profile[36:40] != b"acsp":
        raise HeifPrivacyError("Unsupported ICC profile header")
    count = int.from_bytes(profile[128:132], "big")
    if 132 + count * 12 > len(profile):
        raise HeifPrivacyError("Truncated ICC tag table")
    records = []; names = set()
    for position in range(132, 132 + count * 12, 12):
        kind = profile[position:position + 4]
        offset, length = struct.unpack_from(">II", profile, position + 4)
        if kind in names or kind not in _ICC_FUNCTIONAL | _ICC_DESCRIPTIVE:
            raise HeifPrivacyError(f"Unknown or duplicate ICC tag {kind!r}")
        names.add(kind)
        if offset < 132 + count * 12 or length < 8 or offset + length > len(profile):
            raise HeifPrivacyError("ICC tag escapes its data region")
        if kind in _ICC_FUNCTIONAL and (profile[offset:offset + 4] not in _ICC_TYPES[kind] or any(profile[offset + 4:offset + 8])):
            raise HeifPrivacyError("Unsupported functional ICC tag encoding")
        records.append((kind, offset, length))
    result = bytearray(profile)
    result[4:8] = bytes(4)
    result[24:36] = struct.pack(">6H", 1970, 1, 1, 0, 0, 0)
    result[48:56] = bytes(8)
    result[80:128] = bytes(48)
    kept = []
    for kind, offset, length in records:
        if kind in _ICC_FUNCTIONAL:
            kept.append((kind, offset, length)); continue
        if any(offset < start + size and start < offset + length for other, start, size in records if other in _ICC_FUNCTIONAL):
            raise HeifPrivacyError("ICC identifying text overlaps a color transform")
        if kind not in {b"desc", b"cprt"}:
            continue
        data_type = profile[offset:offset + 4]
        if data_type == b"mluc" and length >= 16:
            empty = b"mluc" + bytes(8) + (12).to_bytes(4, "big")
        elif data_type == b"text" and length >= 9:
            empty = b"text" + bytes(5)
        elif data_type == b"desc" and length >= 90:
            empty = b"desc" + bytes(4) + (1).to_bytes(4, "big") + bytes(78)
        else:
            raise HeifPrivacyError("Unsupported ICC profile description encoding")
        result[offset:offset + length] = empty + bytes(length - len(empty))
        kept.append((kind, offset, length))
    result[128:132] = len(kept).to_bytes(4, "big")
    table = b"".join(kind + struct.pack(">II", offset, length) for kind, offset, length in kept)
    result[132:132 + count * 12] = table + bytes((count - len(kept)) * 12)
    # Zero orphan tag bodies and padding after the compacted table. Shared TRC
    # ranges remain unchanged; only descriptive ranges get rewritten.
    position = 132 + len(kept) * 12
    for start, end in sorted((offset, offset + length) for _, offset, length in kept):
        if position < start:
            result[position:start] = bytes(start - position)
        position = max(position, end)
    if position < len(result):
        result[position:] = bytes(len(result) - position)
    for kind, offset, length in records:
        if kind in _ICC_FUNCTIONAL and profile[offset:offset + length] != result[offset:offset + length]:
            raise HeifPrivacyError("ICC scrub changed a color transform")
    return bytes(result)


def privacy_heif_bytes(data: bytes, original_filename: str = "") -> tuple[bytes, dict[str, Any]]:
    """Scrub a supported still-image container; prove retained payload bytes equal.

    No UUID/name replacement is ever performed by scanning compressed image data.
    Photograph identifiers leave with descriptive metadata. Schema UUIDs and local
    integer relationship IDs remain intact so stereo cameras keep their meaning.
    """
    top, tables, items, locations = _parse(data)
    output = bytearray(data); removed = set(); rewritten = set(); randomized_names = 0; orientation_items = 0
    location_by_id = {location.identifier: location for location in locations}
    ipco = _one(_boxes(data, tables[b"iprp"].payload, tables[b"iprp"].end), b"ipco")
    icc_count = 0
    for property_box in _boxes(data, ipco.payload, ipco.end):
        if property_box.kind != b"colr":
            continue
        color_type = data[property_box.payload:property_box.payload + 4]
        if color_type in {b"prof", b"rICC"}:
            output[property_box.payload + 4:property_box.end] = _private_icc(data[property_box.payload + 4:property_box.end]); icc_count += 1
        elif color_type != b"nclx":
            raise HeifPrivacyError("Unknown color metadata could retain identifying content")
    for identifier, item in items.items():
        if item.kind == b"Exif":
            ranges = location_by_id[identifier].ranges
            minimal = _orientation_exif(b"".join(data[start:end] for start, end in ranges))
            if minimal is None:
                removed.add(identifier)
            else:
                _write_payload(output, ranges, minimal, b"\0"); rewritten.add(identifier); orientation_items += 1
        elif item.kind == b"uri ":
            if item.uri != b"tag:apple.com,2023:photo:metadata:styles":
                raise HeifPrivacyError("Unknown URI metadata could be needed for image functionality")
            removed.add(identifier)
        elif item.kind == b"mime":
            if item.content_type != b"application/rdf+xml":
                raise HeifPrivacyError("Unknown MIME metadata cannot be certified as descriptive")
            ranges = location_by_id[identifier].ranges
            payload = b"".join(data[start:end] for start, end in ranges)
            cleaned = _functional_xmp(payload)
            if cleaned is None:
                removed.add(identifier)
            else:
                _write_payload(output, ranges, cleaned, b" ")
                rewritten.add(identifier)
        if identifier not in removed:
            start, end = item.name_span
            if end > start:
                token = secrets.token_hex((end - start + 1) // 2).encode()[:end - start]
                output[start:end] = token; randomized_names += 1
    # Validate metadata is only descriptive and remove its relationship records.
    if b"iref" in tables:
        _replace_table(output, tables[b"iref"], _references(data, tables[b"iref"], removed, set(items)))
    if b"grpl" in tables:
        box = tables[b"grpl"]
        for group in _boxes(data, box.payload, box.end):
            if group.kind not in {b"altr", b"ster", b"uuid"}:
                raise HeifPrivacyError("Unknown HEIF image grouping")
            if group.kind == b"uuid" and data[group.payload - 16:group.payload] != _SPATIAL_GROUP:
                raise HeifPrivacyError("Unknown UUID group could contain identifying metadata")
            cursor = _full(data, group, {0}, 2 if group.kind == b"uuid" else 0)
            cursor.number(4)
            grouped = [cursor.number(4) for _ in range(cursor.number(4))]; cursor.finish()
            if any(identifier in removed for identifier in grouped):
                raise HeifPrivacyError("Descriptive metadata belongs to a functional image group")
    # Metadata must not share any extent with retained image/function data.
    retained = [(start, end) for location in locations if location.identifier not in removed for start, end in location.ranges]
    for location in locations:
        if location.identifier in removed | rewritten:
            for start, end in location.ranges:
                for other in locations:
                    if other.identifier != location.identifier and any(start < b and a < end for a, b in other.ranges):
                        raise HeifPrivacyError("Metadata extents overlap another HEIF item")
        if location.identifier in removed:
            for start, end in location.ranges:
                output[start:end] = bytes(end - start)
    iinf = tables[b"iinf"]; count_size = 2 if data[iinf.payload] == 0 else 4
    entries = [bytes(output[item.box.start:item.box.end]) for identifier, item in items.items() if identifier not in removed]
    _replace_table(output, iinf, data[iinf.payload:iinf.payload + 4] + len(entries).to_bytes(count_size, "big") + b"".join(entries))
    iloc = tables[b"iloc"]; count_size = 2 if data[iloc.payload] < 2 else 4
    rows = [location.raw for location in locations if location.identifier not in removed]
    _replace_table(output, iloc, data[iloc.payload:iloc.payload + 6] + len(rows).to_bytes(count_size, "big") + b"".join(rows))
    hdlr = tables[b"hdlr"]
    output[hdlr.payload + 24:hdlr.end] = bytes(hdlr.end - hdlr.payload - 24)
    # Clear old metadata/padding not referenced by any retained item. Opaque
    # orphan bytes cannot silently survive in the output's mdat/idat/free spaces.
    payload_boxes = [box for box in top if box.kind in {b"mdat", b"free", b"skip"}]
    if b"idat" in tables:
        payload_boxes.append(tables[b"idat"])
    for box in payload_boxes:
        spans = sorted((max(box.payload, start), min(box.end, end)) for start, end in retained
                       if start < box.end and box.payload < end)
        pos = box.payload
        for start, end in spans:
            if pos < start:
                output[pos:start] = bytes(start - pos)
            pos = max(pos, end)
        if pos < box.end:
            output[pos:box.end] = bytes(box.end - pos)
    # Existing free/skip boxes within the structural containers can also contain
    # obsolete metadata. Reparse after table resizing to include new padding.
    current = bytes(output)
    current_meta = _one(_boxes(current, 0, len(current)), b"meta")
    meta_children = _boxes(current, current_meta.payload + 4, current_meta.end)
    padding = [box for box in meta_children if box.kind in {b"free", b"skip"}]
    properties = _one(meta_children, b"iprp")
    padding.extend(box for box in _boxes(current, properties.payload, properties.end) if box.kind in {b"free", b"skip"})
    for box in padding:
        output[box.payload:box.end] = bytes(box.end - box.payload)
    result = bytes(output)
    _, _, actual_items, actual_locations = _parse(result)
    if set(actual_items) != set(items) - removed:
        raise HeifPrivacyError("Scrubbed HEIF inventory did not round-trip")
    actual_by_id = {location.identifier: location for location in actual_locations}
    preserved = 0
    for identifier, item in items.items():
        if item.kind not in _IMAGES:
            continue
        before = location_by_id[identifier]; after = actual_by_id[identifier]
        if before.ranges != after.ranges or any(data[start:end] != result[start:end] for start, end in before.ranges):
            raise HeifPrivacyError("Scrubbing changed encoded image/auxiliary bytes or their offsets")
        preserved += 1
    if original_filename and original_filename.encode("utf-8") in result:
        raise HeifPrivacyError("Source filename remains in a preserved opaque payload; export refused")
    return result, {"privacy_verified": True, "method": "HEIF metadata item removal with stable byte offsets",
        "compressed_payload_bit_exact": True, "preserved_image_item_count": preserved,
        "removed_metadata_item_count": len(removed), "functional_xmp_item_count": len(rewritten) - orientation_items,
        "orientation_only_exif_items": orientation_items,
        "scrubbed_icc_profiles": icc_count,
        "randomized_item_names": randomized_names, "uuid_map": {},
        "scope": "Exif and descriptive XMP/Apple edit metadata removed; functional color, camera, depth, matte, and HDR descriptions retained. Visible scene content remains identifiable.",
        "sha256": hashlib.sha256(result).hexdigest()}


def scrub_heif(source_path: Path, destination_path: Path) -> dict[str, Any]:
    """Write a new scrubbed candidate exclusively; never modify the source."""
    source_path, destination_path = Path(source_path), Path(destination_path)
    if source_path.resolve() == destination_path.resolve():
        raise HeifPrivacyError("Privacy export requires a new destination")
    data, report = privacy_heif_bytes(source_path.read_bytes(), source_path.name)
    with destination_path.open("xb") as output:
        output.write(data); output.flush(); os.fsync(output.fileno())
    return report
