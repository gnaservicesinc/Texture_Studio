"""Stage only the current material backend dependency closure."""
from pathlib import Path
import shutil
import sys

MATERIAL_SOURCES = (
    "material_workbench.py", "material_model_workbench.py", "material_lora.py",
    "material_dataset.py", "material_resources.py", "material_native_size.py", "material_pbrnxt.py",
    "material_pbrnxt_data.py", "train_material_pbrnxt.py",
)


def stage(root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = set()
    for name in MATERIAL_SOURCES:
        source = root / "scripts" / name
        target = destination / name
        shutil.copy2(source, target)
        expected.add(target)
    shutil.copy2(root / "LICENSE", destination / "LICENSE")
    expected.add(destination / "LICENSE")
    for file in destination.rglob("*"):
        if file.is_file() and file not in expected:
            file.unlink()
    for directory in sorted(destination.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


if __name__ == "__main__":
    stage(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
