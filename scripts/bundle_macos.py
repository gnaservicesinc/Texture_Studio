#!/usr/bin/env python3
"""Deploy and audit the native suite; optionally add a relocatable Python runtime.

The runtime is shared by nested apps. It contains the configured interpreter's
standard library and the pinned runtime dependency closure, never model weights,
datasets, user site-packages or editable links back to this checkout.
"""
from __future__ import annotations

import argparse
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


MACHO_MAGICS = {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
                b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"}
SYSTEM_PREFIXES = ("/System/Library/", "/usr/lib/")


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
    return lines[1] if len(lines) == 2 else None


def rpaths(path: Path) -> list[str]:
    lines = run("otool", "-l", path, capture=True).splitlines()
    return list(dict.fromkeys(lines[index + 2].strip().split("path ", 1)[1].split(" (offset", 1)[0]
                             for index, line in enumerate(lines)
                             if line.strip() == "cmd LC_RPATH"))


def loader_reference(source: Path, destination: Path) -> str:
    return "@loader_path/" + os.path.relpath(destination, source.parent)


def copy_release_packages(destination: Path) -> None:
    """Copy only pinned runtime distributions, including their native assets."""
    lock = Path(__file__).resolve().parents[1] / "requirements-release-macos.txt"
    destination.mkdir(parents=True)
    for line in lock.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, expected = line.split("==", 1)
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected:
            raise RuntimeError(f"Release runtime requires {name}=={expected}; installed {distribution.version}. Run make setup or update and validate the release lock intentionally.")
        package_root = Path(distribution.locate_file("")).resolve()
        if distribution.files is None:
            raise RuntimeError(f"Distribution {name} has no installed file manifest")
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


def copy_runtime(bundle: Path) -> Path:
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


def sign(bundle: Path) -> None:
    # Every install_name_tool edit invalidates its signature. Sign code from
    # the inside out, then nested apps, then the outer application.
    for binary in macho_files(bundle):
        run("codesign", "--force", "--sign", "-", binary, capture=True)
    for application in apps(bundle):
        for framework in sorted(application.glob("Contents/Frameworks/*.framework")):
            run("codesign", "--force", "--sign", "-", framework, capture=True)
        run("codesign", "--force", "--sign", "-", application, capture=True)
    run("codesign", "--verify", "--deep", "--strict", bundle, capture=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--macdeployqt", type=Path)
    parser.add_argument("--output", type=Path, help="Create a fresh distribution copy instead of changing the build bundle")
    parser.add_argument("--bundle-python", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.output:
        output = args.output.resolve()
        if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
            parser.error("Output and input bundles must be separate")
        if output.exists():
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, output, symlinks=True)
        bundle = output
    if not (bundle / "Contents/Info.plist").is_file():
        parser.error("Input must be a complete macOS application bundle")
    if args.audit_only:
        audit(bundle)
        return
    if args.macdeployqt:
        for application in apps(bundle):
            run(args.macdeployqt, application, "-always-overwrite", "-no-strip", "-no-codesign", "-verbose=0")
    wrapper = copy_runtime(bundle) if args.bundle_python else None
    audit(bundle)
    sign(bundle)
    if wrapper:
        # Run from outside the checkout with a clean environment. Importing
        # these exercises the embedded native dependency chains too.
        subprocess.run([str(wrapper), "-c", "import numpy, scipy, torch, torchvision, cv2, pillow_heif, OpenEXR, rawpy, imagecodecs, tifffile, timm, huggingface_hub, safetensors, omegaconf, addict, einops, evo, e3nn, imageio, datasets; print('Bundled Python, teacher and dataset imports passed')"],
                       cwd="/", env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}, check=True)
        metadata = {"python": sys.version, "source_prefix": str(sys.base_prefix),
                    "runtime": "shared relocatable Python prefix", "notarized": False,
                    "dependency_lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "requirements-release-macos.txt").read_bytes()).hexdigest()}
        (bundle / "Contents/Resources/runtime-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Metadata was written after signing; refresh the outer resource seal.
        run("codesign", "--force", "--sign", "-", bundle, capture=True)
        run("codesign", "--verify", "--deep", "--strict", bundle, capture=True)


if __name__ == "__main__":
    main()
