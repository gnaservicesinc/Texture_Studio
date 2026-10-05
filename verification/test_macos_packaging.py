"""Release OS claims must cover every native architecture in the bundle."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import plistlib
import struct
import tempfile
import unittest
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location("ipde_bundle_macos", Path(__file__).resolve().parents[1] / "scripts/bundle_macos.py")
bundle_macos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bundle_macos)


def thin(minimum: tuple[int, int, int], *, endian: str = "<") -> bytes:
    version = (minimum[0] << 16) | (minimum[1] << 8) | minimum[2]
    return (struct.pack(endian + "8I", 0xfeedfacf, 0x0100000c, 0, 2, 1, 24, 0, 0)
            + struct.pack(endian + "6I", 0x32, 24, 1, version, version, 0))


class MacOSPackagingTests(unittest.TestCase):
    def test_universal_library_self_identifier_is_relocated_once(self):
        identifier = "/Library/Frameworks/Python.framework/Versions/3.14/Python"
        listings = (f"Python:\n{identifier}\n",
                    f"Python (architecture x86_64):\n{identifier}\nPython (architecture arm64):\n{identifier}\n")
        for listing in listings:
            with self.subTest(listing=listing), patch.object(bundle_macos, "run", return_value=listing):
                self.assertEqual(bundle_macos.library_id(Path("Python")), identifier)
        with patch.object(bundle_macos, "run", return_value="Executable:\n"):
            self.assertIsNone(bundle_macos.library_id(Path("Executable")))
        with patch.object(bundle_macos, "run", return_value="Library (architecture x86_64):\n/one\nLibrary (architecture arm64):\n/two\n"):
            with self.assertRaisesRegex(RuntimeError, "Inconsistent library install names"):
                bundle_macos.library_id(Path("Library"))

    def test_every_universal_slice_contributes_to_required_os(self):
        arm, intel = thin((14, 0, 0)), thin((15, 1, 0), endian=">")
        offset = 48
        data = (struct.pack(">2I", 0xcafebabe, 2)
                + struct.pack(">5I", 0x0100000c, 0, offset, len(arm), 0)
                + struct.pack(">5I", 0x01000007, 0, offset + len(arm), len(intel), 0)
                + arm + intel)
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "universal"
            binary.write_bytes(data)
            self.assertEqual(bundle_macos.minimum_macos_versions(binary), [(14, 0, 0), (15, 1, 0)])

    def test_bundle_audit_rejects_binary_newer_than_advertised_macos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Fixture.app"
            executable = root / "Contents/MacOS/Fixture"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(thin((15, 0, 0)))
            (root / "Contents/Info.plist").write_bytes(plistlib.dumps({
                "CFBundleExecutable": "Fixture", "LSMinimumSystemVersion": "14.0"}))
            with patch.object(bundle_macos, "dependencies", return_value=[]), \
                 patch.object(bundle_macos, "rpaths", return_value=[]), \
                 patch.object(bundle_macos, "library_id", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "requires macOS 15.0.0, above declared 14.0"):
                    bundle_macos.audit(root)
                executable.write_bytes(thin((14, 0, 0)))
                bundle_macos.audit(root)


if __name__ == "__main__":
    unittest.main()
