"""The native app installer preserves a previous installation on failed swaps."""
from pathlib import Path
import importlib.util
import subprocess
import plistlib
import tempfile
import unittest
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location(
    "texture_studio_install", Path(__file__).resolve().parents[1] / "scripts/install_macos.py")
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)


class NativeInstallTests(unittest.TestCase):
    def test_running_nested_tool_blocks_install_without_stopping_it(self):
        destination = Path("/Applications/Texture Studio.app")
        child = destination / "Contents/Applications/Material Trainer.app/Contents/MacOS/Material Trainer"
        with patch.object(installer.subprocess, "check_output", return_value=f" 124 {child}\n"):
            with self.assertRaises(installer.RunningApplicationError):
                installer.ensure_closed(destination)

    def test_unrelated_app_with_similar_name_does_not_block_install(self):
        with patch.object(installer.subprocess, "check_output", return_value="124 /Applications/Texture Studio.app.backup/Contents/MacOS/Texture Studio\n"):
            installer.ensure_closed(Path("/Applications/Texture Studio.app"))

    def test_bundle_requires_all_four_nonrecursive_role_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Texture Studio.app"
            self.suite_fixture(source)
            installer.validate_bundle(source)
            child = source / "Contents/Applications/Material Trainer.app"
            (child / "Contents/Applications").mkdir()
            with self.assertRaisesRegex(ValueError, "recursively embedded"):
                installer.validate_bundle(source)
            (child / "Contents/Applications").rmdir()
            (child / "Contents/Info.plist").unlink()
            with self.assertRaises(FileNotFoundError):
                installer.validate_bundle(source)

    def test_if_closed_skips_running_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Texture Studio.app"
            self.suite_fixture(source)
            with patch.object(installer.sys, "argv", ["install", str(source), "--destdir", directory, "--if-closed"]), \
                 patch.object(installer.sys, "platform", "darwin"), \
                 patch.object(installer, "install", side_effect=installer.RunningApplicationError("PID 123")):
                installer.main()

    def test_install_validation_refuses_debug_and_unlabeled_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Texture Studio.app"
            self.suite_fixture(source)
            plist = source / "Contents/Info.plist"
            info = plistlib.loads(plist.read_bytes())
            for configuration in ("Debug", None):
                if configuration is None: info.pop("IPDEBuildConfiguration", None)
                else: info["IPDEBuildConfiguration"] = configuration
                plist.write_bytes(plistlib.dumps(info))
                with self.assertRaisesRegex(ValueError, "optimized Release"):
                    installer.validate_bundle(source)

    @staticmethod
    def suite_fixture(source):
        bundles = [(source, "Texture Studio", "org.ipde.texture-studio", None)]
        bundles.extend((source / "Contents/Applications" / f"{name}.app", name, f"org.ipde.material-{role}", role)
                       for role, name in installer.TOOLS.items())
        for bundle, name, identifier, role in bundles:
            executable = bundle / "Contents/MacOS" / name
            executable.parent.mkdir(parents=True)
            executable.write_text("binary")
            executable.chmod(0o755)
            info = {"CFBundleIdentifier": identifier, "CFBundleExecutable": name, "IPDEBuildConfiguration": "Release"}
            if role: info["MaterialToolRole"] = role
            (bundle / "Contents/Info.plist").write_bytes(plistlib.dumps(info))

    def test_signature_failure_keeps_the_installed_app(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = self.fixture(Path(directory))
            with patch.object(installer, "ensure_closed"), \
                 patch.object(installer.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "codesign")):
                with self.assertRaises(subprocess.CalledProcessError):
                    installer.install(source, destination)
            self.assertEqual((destination / "marker").read_text(), "previous")
            self.assertFalse(list(destination.parent.glob(".texture-studio-install-*")))

    def test_failed_swap_restores_the_previous_app(self):
        self.check_failed_swap(rollback_fails=False)

    def test_failed_rollback_retains_the_previous_app_for_recovery(self):
        self.check_failed_swap(rollback_fails=True)

    @staticmethod
    def fixture(root):
        source = root / "Source.app"
        destination = root / "Applications/Texture Studio.app"
        source.mkdir()
        destination.mkdir(parents=True)
        (source / "marker").write_text("replacement")
        (destination / "marker").write_text("previous")
        return source, destination

    def check_failed_swap(self, *, rollback_fails):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = self.fixture(Path(directory))
            rename = Path.rename

            def fail_replacement(path, target):
                if path != destination and path.name == destination.name:
                    raise OSError("cannot install replacement")
                if rollback_fails and path.name == "previous.app":
                    raise OSError("cannot restore previous app")
                return rename(path, target)

            with patch.object(installer, "ensure_closed"), \
                 patch.object(installer.subprocess, "run"), \
                 patch.object(Path, "rename", fail_replacement):
                error = RuntimeError if rollback_fails else OSError
                with self.assertRaises(error):
                    installer.install(source, destination)
            scratch = list(destination.parent.glob(".texture-studio-install-*"))
            if rollback_fails:
                self.assertFalse(destination.exists())
                self.assertEqual(len(scratch), 1)
                self.assertEqual((scratch[0] / "previous.app/marker").read_text(), "previous")
            else:
                self.assertEqual((destination / "marker").read_text(), "previous")
                self.assertFalse(scratch)


if __name__ == "__main__":
    unittest.main()
