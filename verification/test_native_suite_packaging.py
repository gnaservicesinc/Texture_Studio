"""Exercise real nested-code sealing and packaging without building the Swift app."""
from pathlib import Path
import hashlib
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from scripts.stage_material_backend import MATERIAL_SOURCES, stage

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"review": "Material Review", "compare": "Checkpoint Compare",
         "dataset": "Material Dataset", "train": "Material Trainer"}
TOOL_ICONS = {"review": "MaterialReview", "compare": "CheckpointCompare",
              "dataset": "MaterialDataset", "train": "MaterialTrainer"}


class BuildRunDefaultsTests(unittest.TestCase):
    def plan(self, *args):
        result = subprocess.run(["bash", str(ROOT / "script/build_and_run.sh"), "--dry-run", *args],
                                check=True, capture_output=True, text=True)
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_normal_run_and_verify_use_release_with_nested_role_paths(self):
        self.assertEqual(self.plan()["configuration"], "Release")
        plan = self.plan("--verify", "--tool", "train")
        self.assertEqual(plan["configuration"], "Release")
        self.assertTrue(plan["application"].endswith(
            "/Release/Texture Studio.app/Contents/Applications/Material Trainer.app"))

    def test_debug_is_explicit_and_has_separate_product_path(self):
        plan = self.plan("--debug", "--tool", "review")
        self.assertEqual(plan["configuration"], "Debug")
        self.assertIn("/Debug/Texture Studio.app/Contents/Applications/", plan["application"])


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("clang"), "Requires macOS codesign and clang")
class NativeSuitePackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary_directory = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.binary_directory.name) / "tiny-app"
        subprocess.run(["/usr/bin/clang", "-arch", "arm64", "-x", "c", "-", "-o", str(cls.binary)],
                       input="int main(void) { return 0; }", text=True, check=True, capture_output=True)
        cls.icons = Path(cls.binary_directory.name) / "Icons"
        subprocess.run(["xcrun", "swift", str(ROOT / "scripts/generate_app_icons.swift"), str(cls.icons)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.binary_directory.cleanup()

    def fixture(self, directory):
        app = Path(directory) / "Texture Studio.app"
        executable = app / "Contents/MacOS/Texture Studio"
        executable.parent.mkdir(parents=True)
        shutil.copy2(self.binary, executable)
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
            "CFBundleIdentifier": "org.ipde.texture-studio", "CFBundleExecutable": "Texture Studio",
            "CFBundleName": "Texture Studio", "CFBundleDisplayName": "Texture Studio",
            "CFBundleIconFile": "TextureStudio.icns",
            "IPDEBuildConfiguration": "Release",
            "CFBundlePackageType": "APPL", "CFBundleVersion": "1", "CFBundleShortVersionString": "0.9.3"}))
        shutil.copytree(self.icons, app / "Contents/Resources", dirs_exist_ok=True)
        stage(ROOT, app / "Contents/Resources/MaterialBackend")
        entitlements = Path(directory) / "entitlements.plist"
        entitlements.write_bytes(plistlib.dumps({"com.apple.security.network.client": True}))
        subprocess.run(["codesign", "--force", "--sign", "-", "--options", "runtime", "--entitlements",
                        str(entitlements), str(app)], check=True, capture_output=True)
        return app

    def stage(self, app):
        subprocess.run(["bash", str(ROOT / "script/stage_material_apps.sh"), str(app)], check=True,
                       capture_output=True, text=True)

    def test_child_signatures_parent_seal_entitlements_and_nonrecursive_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            self.stage(app)
            self.stage(app)
            children = sorted((app / "Contents/Applications").glob("*.app"))
            self.assertEqual(len(children), 4)
            for role, name in TOOLS.items():
                child = app / "Contents/Applications" / f"{name}.app"
                info = plistlib.loads((child / "Contents/Info.plist").read_bytes())
                self.assertEqual(info["CFBundleIdentifier"], f"org.ipde.material-{role}")
                self.assertEqual(info["MaterialToolRole"], role)
                self.assertEqual(info["CFBundleExecutable"], name)
                self.assertEqual(info["CFBundleIconFile"], f"{TOOL_ICONS[role]}.icns")
                self.assertEqual(sorted(file.name for file in (child / "Contents/Resources").glob("*.icns")),
                                 [f"{TOOL_ICONS[role]}.icns"])
                self.assertFalse((child / "Contents/Applications").exists())
            for application in [app, *children]:
                subprocess.run(["codesign", "--verify", "--deep", "--strict", str(application)], check=True,
                               capture_output=True)
                result = subprocess.run(["codesign", "-d", "--entitlements", "-", "--xml", str(application)],
                                        check=True, capture_output=True)
                self.assertTrue(plistlib.loads(result.stdout)["com.apple.security.network.client"])
            self.assertFalse(list(Path(directory).glob(".material-suite.*")))

    def test_generated_icons_decode_at_every_standard_mac_size(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ["TextureStudio", *TOOL_ICONS.values()]:
                icon = self.icons / f"{name}.icns"
                self.assertLess(icon.stat().st_size, 2 * 1024 * 1024)
                iconset = Path(directory) / f"{name}.iconset"
                subprocess.run(["iconutil", "--convert", "iconset", "--output", str(iconset), str(icon)],
                               check=True, capture_output=True)
                for size in (16, 32, 128, 256, 512):
                    for scale in (1, 2):
                        suffix = "@2x" if scale == 2 else ""
                        image = iconset / f"icon_{size}x{size}{suffix}.png"
                        result = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(image)],
                                                check=True, capture_output=True, text=True)
                        self.assertIn(f"pixelWidth: {size * scale}", result.stdout)
                        self.assertIn(f"pixelHeight: {size * scale}", result.stdout)

    def test_package_contains_four_sealed_children_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            self.stage(app)
            output = Path(directory) / "dist"
            subprocess.run(["bash", str(ROOT / "scripts/package_macos.sh"), str(app), str(output)],
                           check=True, capture_output=True)
            archive = output / "Texture-Studio-macos-arm64.zip"
            with zipfile.ZipFile(archive) as stream:
                names = set(stream.namelist())
            for name in TOOLS.values():
                self.assertIn(f"Texture Studio.app/Contents/Applications/{name}.app/Contents/MacOS/{name}", names)
            self.assertIn("Texture Studio.app/Contents/Resources/TextureStudio.icns", names)
            for application in ["Texture Studio.app", *(
                    f"Texture Studio.app/Contents/Applications/{name}.app" for name in TOOLS.values())]:
                for name in (*MATERIAL_SOURCES, "LICENSE"):
                    self.assertIn(f"{application}/Contents/Resources/MaterialBackend/{name}", names)
            self.assertFalse(any("/DA3Backend/" in name for name in names))
            expected = (output / "Texture-Studio-macos-arm64.sha256").read_text().split()[0]
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)

    def test_packaging_rejects_missing_backend_source_in_parent_or_child(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            self.stage(app)
            output = Path(directory) / "dist"
            command = ["bash", str(ROOT / "scripts/package_macos.sh"), str(app), str(output)]
            applications = [app, *(app / "Contents/Applications" / f"{name}.app" for name in TOOLS.values())]
            for application in applications:
                for name in (*MATERIAL_SOURCES, "LICENSE"):
                    with self.subTest(application=application.name, resource=name):
                        resource = application / "Contents/Resources/MaterialBackend" / name
                        original = resource.read_bytes()
                        resource.unlink()
                        result = subprocess.run(command, capture_output=True, text=True)
                        resource.write_bytes(original)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Missing required material backend resource", result.stderr)
                        self.assertIn(str(resource), result.stderr)
                        self.assertFalse(output.exists())

    def test_packaging_rejects_backend_bytecode_and_runtime_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            self.stage(app)
            output = Path(directory) / "dist"
            command = ["bash", str(ROOT / "scripts/package_macos.sh"), str(app), str(output)]
            for name in ("__pycache__/material_workbench.pyc", "runtime/pyvenv.cfg", "model.safetensors"):
                with self.subTest(resource=name):
                    resource = app / "Contents/Resources/MaterialBackend" / name
                    resource.parent.mkdir(parents=True, exist_ok=True)
                    resource.write_bytes(b"unwanted payload")
                    result = subprocess.run(command, capture_output=True, text=True)
                    resource.unlink()
                    if resource.parent.name != "MaterialBackend":
                        resource.parent.rmdir()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Material backend must contain source and license only", result.stderr)
                    self.assertFalse(output.exists())

    def test_staging_signing_failure_keeps_original_parent_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            before = {str(file.relative_to(app)): hashlib.sha256(file.read_bytes()).hexdigest()
                      for file in app.rglob("*") if file.is_file()}
            result = subprocess.run(["bash", str(ROOT / "script/stage_material_apps.sh"), str(app)],
                                    env={**os.environ, "TEXTURE_STUDIO_SIGN_IDENTITY": "missing-fixture-signing-identity"},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            after = {str(file.relative_to(app)): hashlib.sha256(file.read_bytes()).hexdigest()
                     for file in app.rglob("*") if file.is_file()}
            self.assertEqual(after, before)
            subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True,
                           capture_output=True)
            self.assertFalse(list(Path(directory).glob(".material-suite.*")))

    def test_actual_install_keeps_nested_tool_signatures_and_is_idempotent(self):
        from verification.test_macos_install import installer
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            self.stage(app)
            destination = Path(directory) / "Applications/Texture Studio.app"
            installer.install(app, destination)
            installer.validate_bundle(destination)
            installer.install(app, destination)
            self.assertTrue(installer.same_signed_build(app, destination))
            for name in TOOLS.values():
                relative = Path("Contents/Applications") / f"{name}.app" / "Contents/MacOS" / name
                self.assertEqual((app / relative).read_bytes(), (destination / relative).read_bytes())

    def test_packaging_rejects_missing_child_and_experiment_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            command = ["bash", str(ROOT / "scripts/package_macos.sh"), str(app), str(Path(directory) / "dist")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nested material tool", result.stderr)
            self.stage(app)
            (app / "Contents/Resources/checkpoint.pt").write_bytes(b"model")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("excludes model weights", result.stderr)

    def test_missing_child_icon_does_not_publish_an_incomplete_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.fixture(directory)
            (app / "Contents/Resources/MaterialReview.icns").unlink()
            before = {str(file.relative_to(app)): hashlib.sha256(file.read_bytes()).hexdigest()
                      for file in app.rglob("*") if file.is_file()}
            result = subprocess.run(["bash", str(ROOT / "script/stage_material_apps.sh"), str(app)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing Material Review icon", result.stderr)
            after = {str(file.relative_to(app)): hashlib.sha256(file.read_bytes()).hexdigest()
                     for file in app.rglob("*") if file.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
