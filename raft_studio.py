#!/usr/bin/env python3
"""Standalone RAFT Studio entry point for the source tree and app bundle."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True

source = Path(__file__).resolve().parent / "src"
if source.is_dir():
    sys.path.insert(0, str(source))

from ipde.trainer_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
