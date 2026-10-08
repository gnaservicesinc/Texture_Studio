#!/usr/bin/env python3
"""Keep native, Python and tag versions consistent without importing the app."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--is-prerelease", action="store_true")
    parser.add_argument("--release-notes", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    project = (root / "native/TextureStudio/TextureStudio.xcodeproj/project.pbxproj").read_text()
    native_versions = set(re.findall(r"MARKETING_VERSION = ([0-9.]+);", project))
    if len(native_versions) != 1:
        parser.error(f"Native target versions disagree: {sorted(native_versions)}")
    native = next(iter(native_versions))
    backend = re.search(r'__version__ = "([0-9.]+)"', (root / "src/ipde/__init__.py").read_text()).group(1)
    if version != native or version != backend:
        parser.error(f"Version mismatch: project={version}, native={native}, backend={backend}")
    tag = os.environ.get("TAG", "")
    if os.environ.get("REF_TYPE") == "tag" or args.release_notes:
        if tag != "v" + version:
            parser.error(f"Tag {tag!r} must match source version v{version}")
    prerelease = tuple(int(part) for part in version.split(".")) < (1, 0, 0)
    if args.release_notes:
        notes = f"Texture Studio v{version}\n\n"
        if prerelease:
            notes += "Development pre-release. Save files, datasets, projects and checkpoints may change incompatibly until 1.0.0. Preserve originals and exports before upgrading.\n\n"
        notes += "Requires an Apple Silicon Mac with macOS 26 or later. Download the arm64 ZIP and move Texture Studio.app to Applications. This standalone SwiftUI app uses Apple image, Metal and Core ML backends. Optional models are managed from the Models window. Python extraction and research tools remain available separately in the source repository.\n\n"
        notes += "This release is ad-hoc signed and is not notarized. macOS may require explicit approval in Privacy & Security on first launch.\n\nSee the bundled manual and https://github.com/gnaservicesinc/ipde for setup and issue reporting.\n"
        args.release_notes.write_text(notes)
    if args.is_prerelease:
        sys.exit(0 if prerelease else 1)
    print(f"Version {version}; {'pre-release' if prerelease else 'stable release'}")


if __name__ == "__main__":
    main()
