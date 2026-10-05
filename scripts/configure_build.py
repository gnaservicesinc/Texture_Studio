#!/usr/bin/env python3
"""Run CMake only when the selected kit, interpreter or build options change.

CMake/Ninja already track CMakeLists and included modules. Keeping the requested
configuration separately avoids reconfiguring, timestamp changes and relinking
on every make invocation, including installation under sudo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def configure(build: Path, command: list[str], *, force: bool = False) -> bool:
    state = build / ".ipde-configure.json"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    requested = {"command": command, "revision": revision.stdout.strip()}
    previous = None
    if state.is_file():
        try:
            previous = json.loads(state.read_text())
        except (ValueError, OSError):
            pass
    if not force and previous == requested and (build / "CMakeCache.txt").is_file() and (build / "build.ninja").is_file():
        print(f"Using existing CMake configuration in {build}")
        return False
    subprocess.run(command, check=True)
    state.write_text(json.dumps(requested, indent=2) + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A CMake configure command must follow --")
    configure(args.build_dir, command, force=args.force)


if __name__ == "__main__":
    main()
