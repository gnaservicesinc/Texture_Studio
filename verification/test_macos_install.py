"""The native app installer preserves a previous installation on failed swaps."""
from pathlib import Path
import importlib.util
import subprocess
import tempfile
import unittest
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location(
    "texture_studio_install", Path(__file__).resolve().parents[1] / "scripts/install_macos.py")
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)


class NativeInstallTests(unittest.TestCase):
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
