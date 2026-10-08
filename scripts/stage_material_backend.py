"""Stage local GPL Python sources, never models/datasets, into the Xcode app."""
from pathlib import Path
import shutil
import sys
import tempfile


DA3_METADATA = ("UPSTREAM_LICENSE", "UPSTREAM_REVISION", "requirements.txt")
DA3_ENTRY_POINTS = ("worker.py", "setup_runtime.py")
DA3_REQUIRED_UPSTREAM = (
    "upstream/depth_anything_3/api.py",
    "upstream/depth_anything_3/configs/da3-giant.yaml",
)
EXCLUDED_SOURCE_DIRECTORIES = {
    "__pycache__", ".git", ".venv", "venv", "runtime", "runtimes", "site-packages",
}


def stage_da3(source: Path, destination: Path) -> None:
    """Replace Xcode's folder copy with dependency pins and plain source only.

    Local inference can create bytecode beside the vendored modules. Folder
    resources copy that bytecode (and any accidental model/runtime payloads),
    so select allowed source paths instead of filtering known bad extensions.
    """
    expected = {Path(name): source / name for name in (*DA3_METADATA, *DA3_ENTRY_POINTS)}
    upstream = source / "upstream/depth_anything_3"
    for file in sorted(upstream.rglob("*")):
        relative = file.relative_to(source)
        if file.is_symlink() or not file.is_file():
            continue
        if EXCLUDED_SOURCE_DIRECTORIES.intersection(relative.parts):
            continue
        if file.suffix in {".py", ".yaml", ".yml"}:
            expected[relative] = file
    for name in (*DA3_METADATA, *DA3_ENTRY_POINTS, *DA3_REQUIRED_UPSTREAM):
        file = expected.get(Path(name))
        if file is None or file.is_symlink() or not file.is_file():
            raise FileNotFoundError(f"Missing DA3 source resource: {source / name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Finish copying before replacing the previous build's resources. This
    # also removes stale directories, not just their bytecode files.
    with tempfile.TemporaryDirectory(prefix=".da3-source-stage-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "DA3Backend"
        staged.mkdir()
        for relative, file in sorted(expected.items()):
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        previous = temporary / "previous"
        if destination.exists() or destination.is_symlink():
            destination.rename(previous)
        try:
            staged.rename(destination)
        except OSError:
            if previous.exists() or previous.is_symlink():
                previous.rename(destination)
            raise


def stage(root: Path, destination: Path) -> None:
    stage_da3(root / "native/TextureStudio/Resources/DA3Backend", destination.parent / "DA3Backend")
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
