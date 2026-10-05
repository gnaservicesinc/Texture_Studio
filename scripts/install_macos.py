#!/usr/bin/env python3
"""Install the completed suite after refusing to replace a running installed app."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def ensure_closed(destination: Path) -> None:
    processes = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    for line in processes.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1].startswith(str(destination) + "/"):
            raise RuntimeError(f"Close IPDE Studio and all its subapps before installing (PID {fields[0]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destdir", type=Path, default=Path("/"), help="Staging root; Applications is created below it")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("The install target currently supports macOS only")
    source = args.source.resolve()
    if not (source / "Contents/Resources/python/bin/python3").is_file():
        parser.error("Install requires the completed portable package (run make package)")
    destination = args.destdir.resolve() / "Applications/IPDE Studio.app"
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        parser.error("Source and installed application must be separate")
    ensure_closed(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ipde-install-", dir=destination.parent))
    try:
        staged = staging / destination.name
        shutil.copytree(source, staged, symlinks=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(staged)], check=True)
        ensure_closed(destination)
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
        print(f"Installed {destination}")
    finally:
        shutil.rmtree(staging)


if __name__ == "__main__":
    main()
