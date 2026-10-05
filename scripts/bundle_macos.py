#!/usr/bin/env python3
"""Deploy and audit the native suite; optionally add a relocatable Python runtime.

The runtime is shared by nested apps. It contains the configured interpreter's
standard library and the pinned runtime dependency closure, never model weights,
datasets, user site-packages or editable links back to this checkout.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile


MACHO_MAGICS = {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
                b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"}
SYSTEM_PREFIXES = ("/System/Library/", "/usr/lib/")
RUNTIME_EXCLUDES = {"site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter"}


def clone_or_copy(source: str | Path, destination: str | Path) -> str:
    """Use macOS copy-on-write cloning for large already-built runtimes."""
    source, destination = Path(source), Path(destination)
    if sys.platform == "darwin":
        clonefile = ctypes.CDLL(None, use_errno=True).clonefile
        clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        clonefile.restype = ctypes.c_int
        if clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0:
            shutil.copystat(source, destination)
            return str(destination)
    return shutil.copy2(source, destination)


def runtime_metadata(bundle: Path) -> dict:
    path = bundle / "Contents/Resources/runtime-build.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def locked_distributions():
    lock = Path(__file__).resolve().parents[1] / "requirements-release-macos.txt"
    for line in lock.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, expected = line.split("==", 1)
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected:
            raise RuntimeError(f"Release runtime requires {name}=={expected}; installed {distribution.version}. Run make setup or update and validate the release lock intentionally.")
        if distribution.files is None:
            raise RuntimeError(f"Distribution {name} has no installed file manifest")
        yield distribution


def runtime_inputs(previous: dict | None = None) -> dict:
    """Fingerprint actual runtime inputs without loading costly ML libraries.

    Versions alone miss repaired/replaced files. Size, mtime and ctime cover
    the installed file closure and interpreter library; package manifests and
    the dependency lock ensure additions/removals invalidate the package too.
    """
    base = Path(sys.base_prefix).resolve()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    digest = hashlib.sha256()
    digest.update(sys.version.encode())
    lock = Path(__file__).resolve().parents[1] / "requirements-release-macos.txt"
    digest.update(lock.read_bytes())

    def add(path: Path) -> None:
        digest.update(str(path).encode())
        info = path.stat()
        digest.update(f"{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}:{info.st_mode}".encode())

    for path in (Path(sys.executable).resolve(), base / "Python", base / "LICENSE"):
        if path.is_file():
            add(path)
    stdlib = base / f"lib/python{version}"
    for directory, folders, files in os.walk(stdlib):
        folders[:] = sorted(name for name in folders if name not in RUNTIME_EXCLUDES)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix not in (".a", ".o", ".pyc") and not name.startswith("_tkinter"):
                add(path)
    for distribution in locked_distributions():
        package_root = Path(distribution.locate_file("")).resolve()
        digest.update(f"{distribution.metadata['Name']}=={distribution.version}".encode())
        for record in sorted(distribution.files, key=str):
            relative = Path(str(record))
            if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix in (".pth", ".a", ".o", ".pyc"):
                continue
            original = Path(distribution.locate_file(record))
            if not original.is_file() or not original.resolve().is_relative_to(package_root):
                raise RuntimeError(f"Missing or external installed file for {distribution.metadata['Name']}: {record}")
            add(original)
    external = (previous or {}).get("external_runtime_dependencies", [])
    for name in sorted(external):
        add(Path(name))
    return {"fingerprint": digest.hexdigest(), "external_runtime_dependencies": external}


def write_if_changed(path: Path, value: dict) -> bool:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def sync_tree(source: Path, destination: Path) -> None:
    """Refresh changed native subapp files while retaining unchanged Qt files."""
    destination.mkdir(parents=True, exist_ok=True)
    names = {path.name for path in source.iterdir()}
    for target in destination.iterdir():
        if target.name not in names:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
    for original in source.iterdir():
        target = destination / original.name
        if original.is_symlink():
            link = os.readlink(original)
            if target.is_symlink() and os.readlink(target) == link:
                continue
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(link, target_is_directory=original.is_dir())
        elif original.is_dir():
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                target.unlink()
            sync_tree(original, target)
        else:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            if target.is_symlink():
                target.unlink()
            old = target.stat() if target.exists() else None
            new = original.stat()
            if old is None or (old.st_size, old.st_mtime_ns, old.st_mode) != (new.st_size, new.st_mtime_ns, new.st_mode):
                if target.exists():
                    target.unlink()
                clone_or_copy(original, target)


def run(*arguments: str | Path, capture: bool = False) -> str:
    result = subprocess.run([str(value) for value in arguments], text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        detail = result.stderr or ""
        raise RuntimeError(f"Command failed ({result.returncode}): {arguments!r}\n{detail}")
    return result.stdout if capture else ""


def macho_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) in MACHO_MAGICS:
                result.append(path)
    return sorted(result)


def minimum_macos_versions(binary: Path) -> list[tuple[int, int, int]]:
    """Read every Mach-O slice's minimum OS directly, without executing it."""
    result: list[tuple[int, int, int]] = []
    with binary.open("rb") as stream:
        magic = stream.read(4)
        slices = [0]
        if magic in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
            endian = ">" if magic.startswith(b"\xca") else "<"
            count = struct.unpack(endian + "I", stream.read(4))[0]
            if count > 64:
                raise RuntimeError(f"Invalid Mach-O fat header: {binary}")
            wide = magic in (b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca")
            records = stream.read(count * (32 if wide else 20))
            slices = [struct.unpack_from(endian + ("Q" if wide else "I"), records,
                                         index * (32 if wide else 20) + 8)[0] for index in range(count)]
        for offset in slices:
            stream.seek(offset)
            header = stream.read(28)
            magic = header[:4]
            if magic not in MACHO_MAGICS:
                raise RuntimeError(f"Invalid Mach-O slice: {binary}")
            endian = "<" if magic.startswith(b"\xce") or magic.startswith(b"\xcf") else ">"
            count, size = struct.unpack_from(endian + "II", header, 16)
            if size > 16 * 1024 * 1024 or count > 65536:
                raise RuntimeError(f"Invalid Mach-O commands: {binary}")
            if magic in (b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe"):
                stream.read(4)
            commands = stream.read(size)
            position = 0
            for _ in range(count):
                command, length = struct.unpack_from(endian + "II", commands, position)
                if length < 8 or position + length > len(commands):
                    raise RuntimeError(f"Invalid Mach-O command size: {binary}")
                version = None
                if command == 0x24:  # LC_VERSION_MIN_MACOSX
                    version = struct.unpack_from(endian + "I", commands, position + 8)[0]
                elif command == 0x32:  # LC_BUILD_VERSION, PLATFORM_MACOS
                    platform, minimum = struct.unpack_from(endian + "II", commands, position + 8)
                    if platform == 1:
                        version = minimum
                if version is not None:
                    result.append((version >> 16, (version >> 8) & 255, version & 255))
                position += length
    return result


def dependencies(path: Path) -> list[str]:
    result: list[str] = []
    for line in run("otool", "-L", path, capture=True).splitlines():
        if line.startswith("\t") and " (compatibility version" in line:
            name = line.strip().split(" (compatibility version", 1)[0]
            if name not in result:
                result.append(name)
    return result


def library_id(path: Path) -> str | None:
    lines = run("otool", "-D", path, capture=True).splitlines()
    # Universal binaries repeat the filename/architecture header and install
    # name for each slice. Treat their shared name as one ID, so it can be
    # relocated rather than mistaken for a dependency on the build machine.
    identifiers = {line.strip() for line in lines[1:] if line.strip() and not line.rstrip().endswith(":")}
    if len(identifiers) > 1:
        raise RuntimeError(f"Inconsistent library install names across architectures: {path}")
    return next(iter(identifiers), None)


def rpaths(path: Path) -> list[str]:
    lines = run("otool", "-l", path, capture=True).splitlines()
    return list(dict.fromkeys(lines[index + 2].strip().split("path ", 1)[1].split(" (offset", 1)[0]
                             for index, line in enumerate(lines)
                             if line.strip() == "cmd LC_RPATH"))


def loader_reference(source: Path, destination: Path) -> str:
    return "@loader_path/" + os.path.relpath(destination, source.parent)


def copy_release_packages(destination: Path) -> None:
    """Copy only pinned runtime distributions, including their native assets."""
    destination.mkdir(parents=True)
    for distribution in locked_distributions():
        name = distribution.metadata["Name"]
        package_root = Path(distribution.locate_file("")).resolve()
        for record in distribution.files:
            relative = Path(str(record))
            if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix in (".pth", ".a", ".o", ".pyc"):
                continue
            original = Path(distribution.locate_file(record))
            if not original.is_file() or not original.resolve().is_relative_to(package_root):
                raise RuntimeError(f"Missing or external installed file for {name}: {record}")
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)


def copy_runtime(bundle: Path, external_dependencies: set[str] | None = None) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("Portable macOS packages must be built on macOS")
    base = Path(sys.base_prefix).resolve()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    runtime = bundle / "Contents/Resources/python"
    if runtime.exists():
        shutil.rmtree(runtime)
    prefix = runtime / "prefix"
    prefix.mkdir(parents=True)
    if (base / "Python").is_file():
        shutil.copy2(base / "Python", prefix / "Python")
    (prefix / "bin").mkdir()
    # python.org's bin/python is a small launcher that posix_spawn's the
    # framework's Python.app. Bundle the real interpreter executable instead.
    executable = base / "Resources/Python.app/Contents/MacOS/Python"
    if not executable.is_file():
        executable = base / "bin" / f"python{version}"
    shutil.copy2(executable.resolve(), prefix / "bin" / f"python{version}")
    # Standard-library code is required; unrelated libraries installed beside
    # it are not. Native dependencies are copied on demand below.
    shutil.copytree(base / f"lib/python{version}", prefix / f"lib/python{version}", symlinks=True,
                    ignore=shutil.ignore_patterns("site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "*.a", "*.o"))
    # The suite uses Qt. Exclude Python's optional Tk extension along with its
    # Tk GUI library; it otherwise retains a dependency on the installer Tcl.
    for extension in (prefix / f"lib/python{version}/lib-dynload").glob("_tkinter*"):
        extension.unlink()
    packages = Path(sysconfig.get_path("purelib"))
    destination = prefix / f"lib/python{version}/site-packages"
    copy_release_packages(destination)
    # An environment .pth file may execute code or introduce outside paths.
    # None is needed by the bundled backend, so remove them all.
    for link_file in destination.glob("*.pth"):
        link_file.unlink()
    if (base / "LICENSE").is_file():
        shutil.copy2(base / "LICENSE", runtime / "Python-LICENSE.txt")
    wrapper = runtime / "bin/python3"
    wrapper.parent.mkdir()
    wrapper.write_text(f'''#!/bin/sh
runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
unset PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE PYTHONPLATLIBDIR
export PYTHONHOME="$runtime_dir/prefix"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHONHOME/bin/python{version}" -B "$@"
''')
    wrapper.chmod(0o755)
    external = runtime / "external"
    mappings = [(base, prefix), (packages.resolve(), destination)]
    queue = macho_files(runtime)
    visited: set[Path] = set()
    while queue:
        binary = queue.pop()
        if binary in visited:
            continue
        visited.add(binary)
        own_id = library_id(binary)
        for dependency in dependencies(binary):
            if dependency.startswith(("@loader_path/", "@rpath/")) and dependency != own_id:
                source_binary = None
                for source_root, target_root in reversed(mappings):
                    if binary.is_relative_to(target_root):
                        source_binary = source_root / binary.relative_to(target_root)
                        break
                if source_binary and source_binary.is_file():
                    if dependency.startswith("@loader_path/"):
                        possible = [source_binary.parent / dependency[len("@loader_path/"):]]
                    else:
                        possible = [Path(value.replace("@loader_path", str(source_binary.parent))) / dependency[7:]
                                    for value in rpaths(source_binary) if "@executable_path" not in value]
                    for candidate in possible:
                        candidate = candidate.resolve()
                        if candidate.is_file() and not str(candidate).startswith(SYSTEM_PREFIXES):
                            for source_root, target_root in reversed(mappings):
                                if candidate.is_relative_to(source_root):
                                    copied = target_root / candidate.relative_to(source_root)
                                    if not copied.exists():
                                        copied.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(candidate, copied)
                                        queue.append(copied)
                                    run("install_name_tool", "-change", dependency, loader_reference(binary, copied), binary, capture=True)
                                    break
            if dependency == own_id or not dependency.startswith("/") or dependency.startswith(SYSTEM_PREFIXES):
                continue
            original = Path(dependency)
            copied = None
            # Prefer the specific environment mapping if the environment is
            # inside base_prefix (non-venv release builds).
            for source_root, target_root in reversed(mappings):
                try:
                    copied = target_root / original.relative_to(source_root)
                    break
                except ValueError:
                    pass
            if copied is None:
                if not original.is_file():
                    raise RuntimeError(f"Missing runtime dependency: {binary}: {dependency}")
                if external_dependencies is not None:
                    external_dependencies.add(str(original.resolve()))
                external.mkdir(exist_ok=True)
                digest = hashlib.sha256(str(original).encode()).hexdigest()[:12]
                copied = external / (digest + "-" + original.name)
                if not copied.exists():
                    shutil.copy2(original.resolve(), copied)
                    queue.append(copied)
            if not copied.is_file():
                if not original.is_file():
                    raise RuntimeError(f"Runtime dependency omitted from bundle: {dependency}")
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original.resolve(), copied)
                queue.append(copied)
            run("install_name_tool", "-change", dependency, loader_reference(binary, copied), binary, capture=True)
        if own_id and own_id.startswith("/"):
            run("install_name_tool", "-id", "@rpath/" + binary.name, binary, capture=True)
        for value in rpaths(binary):
            if value.startswith("/") and not value.startswith(SYSTEM_PREFIXES):
                replacement = None
                for source_root, target_root in reversed(mappings):
                    try:
                        replacement = loader_reference(binary, target_root / Path(value).relative_to(source_root))
                        break
                    except ValueError:
                        pass
                run("install_name_tool", "-delete_rpath", value, binary, capture=True)
                if replacement and replacement not in rpaths(binary):
                    run("install_name_tool", "-add_rpath", replacement, binary, capture=True)
    relocate_runtime_rpaths(runtime, prefix / "bin" / f"python{version}")
    return wrapper


def relocate_runtime_rpaths(runtime: Path, executable: Path) -> None:
    """Give extensions explicit paths instead of depending on import order.

    TorchVision wheels expect Torch to have loaded its libraries first. A
    direct reference to the bundled file makes that dependency auditable and
    independent of a build-machine rpath or previously loaded global library.
    """
    binaries = macho_files(runtime)
    executable_paths = rpaths(executable)
    by_name: dict[str, list[Path]] = {}
    for binary in binaries:
        by_name.setdefault(binary.name, []).append(binary)
    for binary in binaries:
        own_id = library_id(binary)
        paths = [*rpaths(binary), *executable_paths]
        for dependency in dependencies(binary):
            if dependency == own_id or not dependency.startswith("@rpath/"):
                continue
            if any((expand_reference(value, binary, executable) / dependency[7:]).is_file() for value in paths):
                continue
            candidates = by_name.get(Path(dependency).name, [])
            if len(candidates) == 1:
                run("install_name_tool", "-change", dependency,
                    loader_reference(binary, candidates[0]), binary, capture=True)


def apps(bundle: Path) -> list[Path]:
    return sorted([bundle, *bundle.rglob("*.app")], key=lambda path: len(path.parts), reverse=True)


def executable_for(binary: Path, bundle: Path) -> Path:
    # Python extension modules may use @executable_path relative to Python,
    # whereas frameworks/plugins belong to the nearest containing native app.
    runtime = bundle / "Contents/Resources/python"
    if binary.is_relative_to(runtime):
        return next(runtime.glob("prefix/bin/python*"))
    for parent in binary.parents:
        info = parent / "Contents/Info.plist"
        if parent.suffix == ".app" and info.exists():
            with info.open("rb") as stream:
                name = plistlib.load(stream)["CFBundleExecutable"]
            return parent / "Contents/MacOS" / name
    raise RuntimeError(f"No executable context for {binary}")


def expand_reference(reference: str, binary: Path, executable: Path) -> Path:
    return Path(reference.replace("@loader_path", str(binary.parent))
                .replace("@executable_path", str(executable.parent)))


def audit(bundle: Path) -> None:
    failures: list[str] = []
    binaries = macho_files(bundle)
    with (bundle / "Contents/Info.plist").open("rb") as stream:
        declared = plistlib.load(stream).get("LSMinimumSystemVersion")
    if not declared:
        raise RuntimeError("Bundle must declare LSMinimumSystemVersion")
    floor = tuple(int(part) for part in declared.split("."))
    floor = (*floor, *(0 for _ in range(3 - len(floor))))
    executable_paths: dict[Path, list[str]] = {}
    for binary in binaries:
        executable = executable_for(binary, bundle)
        for minimum in minimum_macos_versions(binary):
            if minimum > floor:
                failures.append(f"{binary.relative_to(bundle)}: requires macOS {'.'.join(map(str, minimum))}, above declared {declared}")
        own_id = library_id(binary)
        own_paths = rpaths(binary)
        if executable not in executable_paths:
            executable_paths[executable] = own_paths if binary == executable else rpaths(executable)
        paths = [*own_paths, *([] if binary == executable else executable_paths[executable])]
        for value in own_paths:
            if value.startswith("/") and not value.startswith(SYSTEM_PREFIXES):
                failures.append(f"{binary.relative_to(bundle)}: external RPATH {value}")
        for dependency in dependencies(binary):
            if dependency == own_id or dependency.startswith(SYSTEM_PREFIXES):
                continue
            if dependency.startswith("/"):
                failures.append(f"{binary.relative_to(bundle)}: external dependency {dependency}")
                continue
            if dependency.startswith("@rpath/"):
                candidates = [expand_reference(value, binary, executable) / dependency[7:] for value in paths]
            else:
                candidates = [expand_reference(dependency, binary, executable)]
            if not any(candidate.is_file() and candidate.resolve().is_relative_to(bundle.resolve()) for candidate in candidates):
                failures.append(f"{binary.relative_to(bundle)}: unresolved dependency {dependency}")
    if failures:
        raise RuntimeError("Bundle dependency audit failed:\n" + "\n".join(failures))
    print(f"Verified {len(binaries)} native binaries have no external non-system dependencies and support declared macOS {declared}")


def sign(bundle: Path, *, preserved_runtime: bool = False) -> None:
    # Every install_name_tool edit invalidates its signature. Sign code from
    # the inside out, then nested apps, then the outer application.
    runtime = bundle / "Contents/Resources/python"
    binaries = macho_files(bundle)
    timestamps = {binary: (binary.stat().st_atime_ns, binary.stat().st_mtime_ns) for binary in binaries}
    for binary in binaries:
        if preserved_runtime and binary.is_relative_to(runtime):
            continue
        # Qt libraries already carry valid signatures. Avoid replacing them
        # when only Python source or documentation resources changed.
        verified = subprocess.run(["codesign", "--verify", str(binary)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if verified.returncode:
            run("codesign", "--force", "--sign", "-", binary, capture=True)
    for application in apps(bundle):
        for framework in sorted(application.glob("Contents/Frameworks/*.framework")):
            run("codesign", "--force", "--sign", "-", framework, capture=True)
        run("codesign", "--force", "--sign", "-", application, capture=True)
    run("codesign", "--verify", "--deep", "--strict", bundle, capture=True)
    # Sealing application resources updates its Mach-O signature. Retain the
    # linked output's timestamp so it does not invalidate the deployment and
    # resource stamps on the following otherwise unchanged build.
    for binary, (accessed, modified) in timestamps.items():
        if binary.stat().st_mtime_ns != modified:
            os.utime(binary, ns=(accessed, modified))


def check_runtime_imports(bundle: Path) -> None:
    wrapper = bundle / "Contents/Resources/python/bin/python3"
    if not wrapper.is_file():
        raise RuntimeError("Release checks require a packaged Python runtime")
    subprocess.run([str(wrapper), "-c", "import numpy, scipy, torch, torchvision, cv2, pillow_heif, OpenEXR, rawpy, imagecodecs, tifffile, timm, huggingface_hub, safetensors, omegaconf, addict, einops, evo, e3nn, imageio, datasets; print('Bundled Python, teacher and dataset imports passed')"],
                   cwd="/", env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}, check=True)


def package(source: Path, output: Path, *, bundle_python: bool, skip_audit: bool, macdeployqt: Path | None = None) -> None:
    """Stage and sign a replacement, retaining the previous package on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ipde-package-", dir=output.parent))
    staged = staging / output.name
    previous = runtime_metadata(output)
    try:
        shutil.copytree(source, staged, symlinks=True, copy_function=clone_or_copy)
        if macdeployqt:
            for application in apps(staged):
                run(macdeployqt, application, "-no-strip", "-no-codesign", "-verbose=0")
        preserved_runtime = False
        if bundle_python:
            inputs = runtime_inputs(previous)
            runtime = output / "Contents/Resources/python"
            if previous.get("runtime_input_fingerprint") == inputs["fingerprint"] and (runtime / "bin/python3").is_file():
                # The old package's resource seal protects the reused runtime.
                run("codesign", "--verify", "--deep", "--strict", output, capture=True)
                copied_runtime = staged / "Contents/Resources/python"
                if copied_runtime.exists():
                    shutil.rmtree(copied_runtime)
                shutil.copytree(runtime, copied_runtime, symlinks=True, copy_function=clone_or_copy)
                write_if_changed(staged / "Contents/Resources/runtime-build.json", previous)
                preserved_runtime = True
                print("Reusing unchanged packaged Python runtime")
            else:
                external: set[str] = set()
                copy_runtime(staged, external)
                inputs = runtime_inputs({"external_runtime_dependencies": sorted(external)})
                metadata = {"python": sys.version, "source_prefix": str(sys.base_prefix),
                            "runtime": "shared relocatable Python prefix", "notarized": False,
                            "dependency_lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "requirements-release-macos.txt").read_bytes()).hexdigest(),
                            "runtime_input_fingerprint": inputs["fingerprint"],
                            "external_runtime_dependencies": inputs["external_runtime_dependencies"]}
                write_if_changed(staged / "Contents/Resources/runtime-build.json", metadata)
        if not skip_audit:
            audit(staged)
        sign(staged, preserved_runtime=preserved_runtime)
        if bundle_python and not preserved_runtime:
            # Validate a newly relocated runtime once; app-only updates reuse
            # these results. Full dependency/import checks remain explicit.
            check_runtime_imports(staged)
        backup = staging / "previous.app"
        if output.exists():
            output.rename(backup)
        try:
            staged.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
    finally:
        shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--macdeployqt", type=Path)
    parser.add_argument("--output", type=Path, help="Update a distribution copy, reusing its unchanged Python runtime")
    parser.add_argument("--bundle-python", action="store_true")
    parser.add_argument("--assemble", nargs="+", type=Path, help="Refresh these subapps inside the suite")
    parser.add_argument("--skip-audit", action="store_true", help="Local build: verify signatures; run release-check for the full dependency audit")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--release-check", action="store_true")
    parser.add_argument("--runtime-inputs", type=Path, help="Update an input fingerprint only if the configured runtime has changed")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.runtime_inputs:
        changed = write_if_changed(args.runtime_inputs, runtime_inputs(runtime_metadata(bundle)))
        print("Python runtime inputs changed" if changed else "Python runtime inputs unchanged")
        return
    if not (bundle / "Contents/Info.plist").is_file():
        parser.error("Input must be a complete macOS application bundle")
    if args.audit_only or args.release_check:
        audit(bundle)
        if args.release_check:
            run("codesign", "--verify", "--deep", "--strict", bundle, capture=True)
            check_runtime_imports(bundle)
        return
    if args.output:
        output = args.output.resolve()
        if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
            parser.error("Output and input bundles must be separate")
        package(bundle, output, bundle_python=args.bundle_python, skip_audit=args.skip_audit,
                macdeployqt=args.macdeployqt)
        return
    if args.assemble:
        destination = bundle / "Contents/Applications"
        for application in args.assemble:
            sync_tree(application.resolve(), destination / application.name)
    if args.macdeployqt:
        for application in apps(bundle):
            run(args.macdeployqt, application, "-no-strip", "-no-codesign", "-verbose=0")
    if args.bundle_python:
        external: set[str] = set()
        copy_runtime(bundle, external)
        inputs = runtime_inputs({"external_runtime_dependencies": sorted(external)})
        write_if_changed(bundle / "Contents/Resources/runtime-build.json", {
            "python": sys.version, "source_prefix": str(sys.base_prefix),
            "runtime": "shared relocatable Python prefix", "notarized": False,
            "runtime_input_fingerprint": inputs["fingerprint"],
            "external_runtime_dependencies": inputs["external_runtime_dependencies"]})
    if not args.skip_audit:
        audit(bundle)
    sign(bundle)
    if args.bundle_python:
        check_runtime_imports(bundle)


if __name__ == "__main__":
    main()
