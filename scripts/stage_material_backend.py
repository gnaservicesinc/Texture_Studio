"""Stage local GPL Python sources, never models/datasets, into the Xcode app."""
from pathlib import Path
import shutil
import sys


def stage(root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = set()
    for source in (root / "scripts").glob("*.py"):
        target = destination / source.name
        shutil.copy2(source, target)
        expected.add(target)
    for source in (root / "src/ipde").rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        target = destination / "ipde" / source.relative_to(root / "src/ipde")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected.add(target)
    shutil.copy2(root / "LICENSE", destination / "LICENSE")
    expected.add(destination / "LICENSE")
    for file in destination.rglob("*"):
        if file.is_file() and file not in expected:
            file.unlink()


if __name__ == "__main__":
    stage(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
