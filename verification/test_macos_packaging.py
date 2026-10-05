"""Release OS claims must cover every native architecture in the bundle."""
from __future__ import annotations

import importlib.util
import errno
import os
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
    def test_package_cleanup_retries_finder_store_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "Source.app", root / "Release.app"
            source.mkdir()
            output.mkdir()
            (source / "marker").write_text("replacement")
            (output / "marker").write_text("previous")
            real_rmdir = os.rmdir
            recreated = []

            def finder_rmdir(path, *args, **kwargs):
                if Path(path).name.startswith(".ipde-package-") and len(recreated) < 2:
                    (Path(path) / ".DS_Store").write_bytes(b"Finder metadata")
                    recreated.append(Path(path))
                return real_rmdir(path, *args, **kwargs)

            with patch.object(bundle_macos, "sign"), \
                 patch.object(bundle_macos.os, "rmdir", side_effect=finder_rmdir), \
                 patch.object(bundle_macos.time, "sleep") as sleep:
                bundle_macos.package(source, output, bundle_python=False, skip_audit=True)
            self.assertEqual((output / "marker").read_text(), "replacement")
            self.assertEqual(len(recreated), 2)
            self.assertEqual(sleep.call_count, 2)
            self.assertFalse(list(root.glob(".ipde-package-*")))

    def test_staging_cleanup_retries_are_bounded(self):
        failure = OSError(errno.ENOTEMPTY, "directory keeps changing")
        with patch.object(bundle_macos.shutil, "rmtree", side_effect=failure) as remove, \
             patch.object(bundle_macos.time, "sleep") as sleep:
            with self.assertRaises(OSError) as raised:
                bundle_macos.remove_package_staging(Path("scratch"))
        self.assertIs(raised.exception, failure)
        self.assertEqual(remove.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_staging_cleanup_propagates_other_errors_immediately(self):
        failure = OSError(errno.EACCES, "permission denied")
        with patch.object(bundle_macos.shutil, "rmtree", side_effect=failure) as remove, \
             patch.object(bundle_macos.time, "sleep") as sleep:
            with self.assertRaises(OSError) as raised:
                bundle_macos.remove_package_staging(Path("scratch"))
        self.assertIs(raised.exception, failure)
        remove.assert_called_once()
        sleep.assert_not_called()

    def test_failed_package_rollback_retains_prior_app(self):
        self.check_failed_replacement(rollback_fails=True)

    def test_successful_package_rollback_restores_prior_app_and_cleans_scratch(self):
        self.check_failed_replacement(rollback_fails=False)

    def check_failed_replacement(self, *, rollback_fails):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "Source.app", root / "Release.app"
            source.mkdir()
            output.mkdir()
            (source / "marker").write_text("replacement")
            (output / "marker").write_text("previous")
            real_rename = Path.rename

            def failed_rename(path, target):
                if path != output and path.name == output.name:
                    raise OSError(errno.ENOSPC, "cannot install replacement")
                if rollback_fails and path.name == "previous.app":
                    raise OSError(errno.EACCES, "cannot restore prior package")
                return real_rename(path, target)

            with patch.object(bundle_macos, "sign"), patch.object(Path, "rename", failed_rename):
                if rollback_fails:
                    with self.assertRaisesRegex(RuntimeError, "previous package retained at"):
                        bundle_macos.package(source, output, bundle_python=False, skip_audit=True)
                else:
                    with self.assertRaises(OSError) as raised:
                        bundle_macos.package(source, output, bundle_python=False, skip_audit=True)
                    self.assertEqual(raised.exception.errno, errno.ENOSPC)
            scratch = list(root.glob(".ipde-package-*"))
            if rollback_fails:
                self.assertEqual(len(scratch), 1)
                self.assertEqual((scratch[0] / "previous.app/marker").read_text(), "previous")
                self.assertEqual((scratch[0] / output.name / "marker").read_text(), "replacement")
                self.assertFalse(output.exists())
            else:
                self.assertEqual((output / "marker").read_text(), "previous")
                self.assertFalse(scratch)

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
