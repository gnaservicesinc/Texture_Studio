#!/usr/bin/env python3
"""Install Texture Studio after refusing to replace a running installed app."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile

TOOLS = {"review": "Material Review", "compare": "Checkpoint Compare",
         "dataset": "Material Dataset", "train": "Material Trainer"}


class RunningApplicationError(RuntimeError):
    pass


def validate_bundle(source: Path) -> None:
    """Require one self-contained parent with all four directly launchable tools."""
    apps = [(source, "org.ipde.texture-studio", "Texture Studio", None)]
    apps.extend((source / "Contents/Applications" / f"{name}.app", f"org.ipde.material-{role}", name, role)
                for role, name in TOOLS.items())
    for app, identifier, name, role in apps:
        info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        if info.get("IPDEBuildConfiguration") != "Release":
            raise ValueError(f"Install requires an optimized Release build: {app}")
        if info.get("CFBundleIdentifier") != identifier or info.get("CFBundleExecutable") != name:
            raise ValueError(f"Unexpected application identity: {app}")
        executable = app / "Contents/MacOS" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Application executable is missing: {executable}")
        if role and (info.get("MaterialToolRole") != role or (app / "Contents/Applications").exists()):
            raise ValueError(f"Invalid or recursively embedded material tool: {app}")


def same_signed_build(source: Path, destination: Path) -> bool:
    """Compare the sealed app code/resources, then verify the installed copy."""
    try:
        with (source / "Contents/Info.plist").open("rb") as stream:
            executable = plistlib.load(stream)["CFBundleExecutable"]
        for name in ("Contents/Info.plist", "Contents/_CodeSignature/CodeResources", f"Contents/MacOS/{executable}"):
            if (source / name).read_bytes() != (destination / name).read_bytes():
                return False
        for application in (source, destination):
            result = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(application)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def ensure_closed(destination: Path) -> None:
    processes = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    for line in processes.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1].startswith(str(destination) + "/"):
            raise RunningApplicationError(f"Close Texture Studio and its material tools before installing (PID {fields[0]})")


def removable_tree(path: Path) -> bool:
    """Check the whole old tree before deleting any of its recoverable files."""
    if path.is_symlink():
        return True
    blocked_flags = sum(getattr(stat, name, 0) for name in ("UF_IMMUTABLE", "SF_IMMUTABLE", "UF_APPEND", "SF_APPEND"))

    def unreadable(error: OSError) -> None:
        raise error

    try:
        for directory, folders, files in os.walk(path, onerror=unreadable):
            current = Path(directory)
            if not os.access(current, os.R_OK | os.W_OK | os.X_OK):
                return False
            info = current.stat()
            if info.st_mode & stat.S_ISVTX and info.st_uid != os.geteuid() and os.geteuid() != 0:
                return False
            for entry in (current, *(current / name for name in (*folders, *files))):
                if getattr(entry.lstat(), "st_flags", 0) & blocked_flags:
                    return False
        return True
    except OSError:
        return False


def install(source: Path, destination: Path) -> None:
    if same_signed_build(source, destination):
        print(f"Already current: {destination}")
        return
    ensure_closed(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".texture-studio-install-", dir=destination.parent))
    backup = staging / "previous.app"
    preserve_backup = False
    try:
        staged = staging / destination.name
        shutil.copytree(source, staged, symlinks=True, copy_function=shutil.copy2)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(staged)], check=True)
        ensure_closed(destination)
        if destination.exists() or destination.is_symlink():
            preserve_backup = True
            destination.rename(backup)
        try:
            staged.rename(destination)
        except BaseException as swap_error:
            if backup.exists() or backup.is_symlink():
                try:
                    backup.rename(destination)
                    preserve_backup = False
                except BaseException as rollback_error:
                    preserve_backup = True
                    raise RuntimeError(f"Installation failed ({swap_error}); restoring the previous app also failed ({rollback_error}). Previous app retained at {backup}") from rollback_error
            raise
        preserve_backup = False
        print(f"Installed {destination}")
    finally:
        if preserve_backup and (backup.exists() or backup.is_symlink()):
            print(f"Previous app retained at {backup}; installation could not restore it.")
        elif (backup.exists() or backup.is_symlink()) and not removable_tree(backup):
            print(f"Previous app retained at {backup}; this account cannot remove its files.")
        else:
            try:
                shutil.rmtree(staging)
            except OSError as error:
                retained = backup if backup.exists() or backup.is_symlink() else staging
                print(f"Installation temporary files retained at {retained}; cleanup failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destdir", type=Path, default=Path("/"), help="Staging root; Applications is created below it")
    parser.add_argument("--if-closed", action="store_true", help="Leave a running installation untouched and report a skipped install")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("The install target currently supports macOS only")
    source = args.source.resolve()
    try:
        validate_bundle(source)
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"Install requires a complete Texture Studio bundle: {error}")
    destination = args.destdir.resolve() / "Applications/Texture Studio.app"
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        parser.error("Source and installed application must be separate")
    try:
        install(source, destination)
    except RunningApplicationError as error:
        if not args.if_closed:
            raise
        print(f"Installation skipped; running application unchanged: {error}")


if __name__ == "__main__":
    main()
