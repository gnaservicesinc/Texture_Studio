"""Verify the same source-only material backend closure staged by Xcode."""
import argparse
from pathlib import Path

from stage_material_backend import MATERIAL_SOURCES


def verify(destination: Path) -> None:
    expected = (*MATERIAL_SOURCES, "LICENSE")
    for name in expected:
        resource = destination / name
        if resource.is_symlink() or not resource.is_file():
            raise ValueError(f"Missing required material backend resource: {resource}")
        if resource.stat().st_size == 0:
            raise ValueError(f"Empty required material backend resource: {resource}")
    for resource in sorted(destination.iterdir()):
        if resource.name not in expected:
            raise ValueError(f"Material backend must contain source and license only: {resource}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        verify(args.destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
