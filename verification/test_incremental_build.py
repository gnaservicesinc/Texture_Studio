"""Local builds reuse configuration and runtime without installing stale code."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import plistlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


configure_build = load_script("configure_build")
bundle_macos = load_script("bundle_macos")
with patch.dict(bundle_macos.sys.modules, {"bundle_macos": bundle_macos}):
    install_macos = load_script("install_macos")


class IncrementalBuildTests(unittest.TestCase):
    def installation_fixture(self, root):
        source = root / "Source.app"
        destination = root / "Applications/IPDE Studio.app"
        source.mkdir()
        destination.mkdir(parents=True)
        (source / "version").write_text("new")
        (destination / "version").write_text("old")
        return source, destination

    def test_install_retains_unremovable_previous_app_without_partial_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = self.installation_fixture(root)
            (destination / "Contents").mkdir()
            (destination / "Contents/data").write_text("complete previous data")
            output = io.StringIO()
            access = install_macos.os.access

            def permitted(path, mode):
                if Path(path).name == "Contents" and "previous.app" in Path(path).parts:
                    return False
                return access(path, mode)

            with patch.object(install_macos, "same_signed_build", return_value=False), \
                 patch.object(install_macos, "ensure_closed"), \
                 patch.object(install_macos.subprocess, "run"), \
                 patch.object(install_macos.os, "access", side_effect=permitted), \
                 patch.object(install_macos.shutil, "rmtree", wraps=install_macos.shutil.rmtree) as remove, \
                 contextlib.redirect_stdout(output):
                install_macos.install(source, destination)
                remove.assert_not_called()
            backup = next(destination.parent.glob(".ipde-install-*/previous.app"))
            self.assertEqual((destination / "version").read_text(), "new")
            self.assertEqual((backup / "version").read_text(), "old")
            self.assertEqual((backup / "Contents/data").read_text(), "complete previous data")
            self.assertIn("Installed " + str(destination), output.getvalue())
            self.assertIn("Previous app retained at " + str(backup), output.getvalue())

    def test_successful_install_reports_cleanup_failure_without_failing(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = self.installation_fixture(Path(directory))
            output = io.StringIO()
            with patch.object(install_macos, "same_signed_build", return_value=False), \
                 patch.object(install_macos, "ensure_closed"), \
                 patch.object(install_macos.subprocess, "run"), \
                 patch.object(install_macos.shutil, "rmtree", side_effect=PermissionError("cleanup denied")), \
                 contextlib.redirect_stdout(output):
                install_macos.install(source, destination)
            backup = next(destination.parent.glob(".ipde-install-*/previous.app"))
            self.assertEqual((destination / "version").read_text(), "new")
            self.assertEqual((backup / "version").read_text(), "old")
            self.assertIn(str(backup), output.getvalue())
            self.assertIn("cleanup denied", output.getvalue())

    def test_failed_install_restores_previous_app_and_cleans_staged_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = self.installation_fixture(Path(directory))
            rename = Path.rename

            def replace(path, target):
                if path.name == destination.name and path.parent.name.startswith(".ipde-install-"):
                    raise PermissionError("replacement denied")
                return rename(path, target)

            with patch.object(install_macos, "same_signed_build", return_value=False), \
                 patch.object(install_macos, "ensure_closed"), \
                 patch.object(install_macos.subprocess, "run"), \
                 patch.object(Path, "rename", replace):
                with self.assertRaisesRegex(PermissionError, "replacement denied"):
                    install_macos.install(source, destination)
            self.assertEqual((destination / "version").read_text(), "old")
            self.assertEqual(list(destination.parent.glob(".ipde-install-*")), [])

    def test_failed_rollback_keeps_both_the_previous_app_and_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = self.installation_fixture(Path(directory))
            rename = Path.rename
            output = io.StringIO()

            def replace(path, target):
                if path.parent.name.startswith(".ipde-install-"):
                    if path.name == "previous.app":
                        raise PermissionError("rollback denied")
                    raise PermissionError("replacement denied")
                return rename(path, target)

            with patch.object(install_macos, "same_signed_build", return_value=False), \
                 patch.object(install_macos, "ensure_closed"), \
                 patch.object(install_macos.subprocess, "run"), \
                 patch.object(Path, "rename", replace), \
                 patch.object(install_macos.shutil, "rmtree") as remove, \
                 contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "replacement denied.*rollback denied.*Previous app retained at"):
                    install_macos.install(source, destination)
                remove.assert_not_called()
            backup = next(destination.parent.glob(".ipde-install-*/previous.app"))
            self.assertEqual((backup / "version").read_text(), "old")
            self.assertEqual((backup.parent / destination.name / "version").read_text(), "new")
            self.assertFalse(destination.exists())
            self.assertIn(str(backup), output.getvalue())

    def test_verified_successful_install_removes_writable_previous_app(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = self.installation_fixture(Path(directory))
            with patch.object(install_macos, "same_signed_build", return_value=False), \
                 patch.object(install_macos, "ensure_closed") as closed, \
                 patch.object(install_macos.subprocess, "run") as verify:
                install_macos.install(source, destination)
                self.assertEqual(closed.call_count, 2)
                self.assertEqual(verify.call_args.args[0][:4], ["codesign", "--verify", "--deep", "--strict"])
            self.assertEqual((destination / "version").read_text(), "new")
            self.assertEqual(list(destination.parent.glob(".ipde-install-*")), [])

    def test_identical_valid_installed_build_is_recognized_and_damaged_copy_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            apps = [Path(directory) / name for name in ("Source.app", "Installed.app")]
            for app in apps:
                (app / "Contents/MacOS").mkdir(parents=True)
                (app / "Contents/_CodeSignature").mkdir()
                (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": "Fixture"}))
                (app / "Contents/MacOS/Fixture").write_bytes(b"signed executable")
                (app / "Contents/_CodeSignature/CodeResources").write_bytes(b"sealed resources")
            with patch.object(install_macos.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as verify:
                self.assertTrue(install_macos.same_signed_build(*apps))
                self.assertEqual(verify.call_count, 2)
            with patch.object(install_macos.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
                self.assertFalse(install_macos.same_signed_build(*apps))
            (apps[1] / "Contents/MacOS/Fixture").write_bytes(b"different build")
            with patch.object(install_macos.subprocess, "run") as verify:
                self.assertFalse(install_macos.same_signed_build(*apps))
                verify.assert_not_called()

    def test_unchanged_configuration_is_reused_and_option_change_reconfigures(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").touch()
            (build / "build.ninja").touch()
            command = ["cmake", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
            with patch.object(configure_build.subprocess, "run", return_value=SimpleNamespace(stdout="revision\n")) as run:
                self.assertTrue(configure_build.configure(build, command))
                self.assertFalse(configure_build.configure(build, command))
                self.assertEqual([call.args[0] for call in run.call_args_list].count(command), 1)
                changed = [*command[:-1], "-DCMAKE_BUILD_TYPE=Debug"]
                self.assertTrue(configure_build.configure(build, changed))
                self.assertTrue(configure_build.configure(build, changed, force=True))
                (build / "build.ninja").unlink()
                self.assertTrue(configure_build.configure(build, changed))

    def test_runtime_fingerprint_detects_same_version_package_and_interpreter_repairs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            python = base / "Python"
            python.write_bytes(b"interpreter")
            module = base / "lib" / f"python{bundle_macos.sys.version_info.major}.{bundle_macos.sys.version_info.minor}" / "fixture.py"
            module.parent.mkdir(parents=True)
            module.write_text("original")
            package = base / "packages/fixture.py"
            package.parent.mkdir()
            package.write_text("original")
            distribution = SimpleNamespace(version="1", metadata={"Name": "fixture"}, files=[Path("fixture.py")], locate_file=lambda path: package.parent / path)
            with patch.object(bundle_macos.sys, "base_prefix", str(base)), \
                 patch.object(bundle_macos.sys, "executable", str(python)), \
                 patch.object(bundle_macos, "locked_distributions", return_value=[distribution]):
                first = bundle_macos.runtime_inputs()
                self.assertEqual(first, bundle_macos.runtime_inputs())
                package.write_text("repaired")
                second = bundle_macos.runtime_inputs()
                self.assertNotEqual(first["fingerprint"], second["fingerprint"])
                python.write_bytes(b"repaired interpreter")
                self.assertNotEqual(second["fingerprint"], bundle_macos.runtime_inputs()["fingerprint"])

    def test_app_update_reuses_runtime_and_does_not_repeat_release_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Source.app"
            output = root / "dist/IPDE Studio.app"
            (source / "Contents/Resources").mkdir(parents=True)
            (source / "Contents/Resources/backend.py").write_text("new implementation")
            runtime = output / "Contents/Resources/python"
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin/python3").write_text("original interpreter")
            metadata = {"runtime_input_fingerprint": "current", "external_runtime_dependencies": []}
            bundle_macos.write_if_changed(output / "Contents/Resources/runtime-build.json", metadata)
            with patch.object(bundle_macos, "runtime_inputs", return_value={"fingerprint": "current"}), \
                 patch.object(bundle_macos, "copy_runtime") as rebuild, \
                 patch.object(bundle_macos, "run") as run, \
                 patch.object(bundle_macos, "sign") as sign, \
                 patch.object(bundle_macos, "audit") as audit, \
                 patch.object(bundle_macos, "check_runtime_imports") as imports:
                bundle_macos.package(source, output, bundle_python=True, skip_audit=True)
                rebuild.assert_not_called()
                audit.assert_not_called()
                imports.assert_not_called()
                self.assertTrue(sign.call_args.kwargs["preserved_runtime"])
                self.assertEqual(run.call_args.args[:4], ("codesign", "--verify", "--deep", "--strict"))
            self.assertEqual((runtime / "bin/python3").read_text(), "original interpreter")
            self.assertEqual((output / "Contents/Resources/backend.py").read_text(), "new implementation")
            self.assertEqual(bundle_macos.runtime_metadata(output), metadata)

    def test_failed_package_keeps_existing_installable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "Source.app", root / "dist/Output.app"
            source.mkdir()
            output.mkdir(parents=True)
            (output / "previous").write_text("usable")
            with patch.object(bundle_macos, "sign", side_effect=RuntimeError("signature failed")):
                with self.assertRaisesRegex(RuntimeError, "signature failed"):
                    bundle_macos.package(source, output, bundle_python=False, skip_audit=True)
            self.assertEqual((output / "previous").read_text(), "usable")
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_runtime_change_rebuilds_and_checks_the_new_runtime_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "Source.app", root / "dist/Output.app"
            (source / "Contents/Resources").mkdir(parents=True)
            (output / "Contents/Resources/python/bin").mkdir(parents=True)
            (output / "Contents/Resources/python/bin/python3").write_text("old interpreter")
            bundle_macos.write_if_changed(output / "Contents/Resources/runtime-build.json", {
                "runtime_input_fingerprint": "old", "external_runtime_dependencies": []})

            def rebuild(staged, external):
                wrapper = staged / "Contents/Resources/python/bin/python3"
                wrapper.parent.mkdir(parents=True)
                wrapper.write_text("new interpreter")
                return wrapper

            with patch.object(bundle_macos, "runtime_inputs", return_value={"fingerprint": "new", "external_runtime_dependencies": []}), \
                 patch.object(bundle_macos, "copy_runtime", side_effect=rebuild) as copied, \
                 patch.object(bundle_macos, "sign") as sign, \
                 patch.object(bundle_macos, "check_runtime_imports") as checked:
                bundle_macos.package(source, output, bundle_python=True, skip_audit=True)
                copied.assert_called_once()
                checked.assert_called_once()
                self.assertFalse(sign.call_args.kwargs["preserved_runtime"])
            self.assertEqual((output / "Contents/Resources/python/bin/python3").read_text(), "new interpreter")
            self.assertEqual(bundle_macos.runtime_metadata(output)["runtime_input_fingerprint"], "new")

    def test_default_make_builds_and_install_keeps_release_audit_explicit(self):
        default = subprocess.run(["make", "-n"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
        self.assertIn("configure_build.py", default)
        self.assertNotIn("-m venv", default)
        self.assertNotIn("-m pip", default)
        install = subprocess.run(["make", "-n", "install"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
        self.assertIn("--target distribution", install)
        self.assertIn("install_macos.py", install)
        self.assertNotIn("check_release_version", install)
        self.assertNotIn("--target release-check", install)


if __name__ == "__main__":
    unittest.main()
