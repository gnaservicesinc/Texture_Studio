"""Development Qt upgrades must select an honest deployment floor."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("cmake"), "CMake is required for deployment selection")
class MacOSDeploymentTests(unittest.TestCase):
    def select(self, qt_minimum, target=""):
        module = Path(__file__).resolve().parents[1] / "cmake/MacOSDeployment.cmake"
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "deployment.cmake"
            script.write_text(f'''include("{module.as_posix()}")
set(QT_SUPPORTED_MIN_MACOS_VERSION "{qt_minimum}")
set(Qt6_VERSION "fixture")
set(CMAKE_OSX_DEPLOYMENT_TARGET "{target}")
ipde_select_macos_deployment(selected)
message(STATUS "Selected=${{selected}}")
''')
            return subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)

    def test_auto_covers_python_and_current_qt_minimum(self):
        for qt, expected in (("13.0", "14.0"), ("14.4", "14.4"), ("15.2", "15.2")):
            with self.subTest(qt=qt):
                result = self.select(qt)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Selected=" + expected, result.stdout)

    def test_explicit_release_and_newer_targets_are_preserved(self):
        for qt, target in (("13.0", "14.0"), ("14.4", "15.0")):
            result = self.select(qt, target)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Selected=" + target, result.stdout)

    def test_incompatible_explicit_target_fails_before_linking(self):
        for qt, target, required in (("14.4", "14.0", "14.4"), ("13.0", "13.5", "14.0")):
            result = self.select(qt, target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("below the required " + required, result.stderr)
            self.assertIn("automatic selection", result.stderr)


if __name__ == "__main__":
    unittest.main()
