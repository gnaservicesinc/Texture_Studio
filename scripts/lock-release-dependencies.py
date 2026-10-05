#!/usr/bin/env python3
"""Freeze the installed macOS release dependency closure after validation.

Run intentionally when upgrading release dependencies. This reads installed
distribution metadata; it never installs packages or includes unrelated ones.
"""
from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    seeds = ["datasets>=4,<6"]
    for filename in ("requirements.txt", "requirements-depth.txt"):
        seeds.extend(line.split("#", 1)[0].strip() for line in (root / filename).read_text().splitlines()
                     if line.split("#", 1)[0].strip())
    environments = []
    for version in ("3.13", "3.14"):
        environment = default_environment()
        environment.update(python_version=version, python_full_version=version + ".0",
                           sys_platform="darwin", platform_machine="arm64", platform_system="Darwin")
        environments.append(environment)
    selected: dict[str, str] = {}
    visited: set[tuple[str, tuple[str, ...]]] = set()
    while seeds:
        requirement = Requirement(seeds.pop())
        name = canonicalize_name(requirement.name)
        distribution = metadata.distribution(requirement.name)
        if not requirement.specifier.contains(distribution.version):
            raise RuntimeError(f"Installed {name}=={distribution.version} does not satisfy {requirement}")
        selected[name] = distribution.version
        key = (name, tuple(sorted(requirement.extras)))
        if key in visited:
            continue
        visited.add(key)
        for expression in distribution.requires or []:
            child = Requirement(expression)
            if not child.marker or any(child.marker.evaluate(dict(environment, extra=extra))
                                       for environment in environments for extra in ("", *requirement.extras)):
                seeds.append(str(child))
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    header = f"""# IPDE Studio v{version} macOS runtime dependency closure.
# Exact versions validated in the development environment; shared by setup,
# the release workflow and the standalone runtime builder. Includes only the
# base/teacher/datasets dependency closure, not unrelated environment packages.
# Regenerate with scripts/lock-release-dependencies.py intentionally when
# updating dependencies, then run all checks before publishing.
"""
    (root / "requirements-release-macos.txt").write_text(header + "\n".join(
        f"{name}=={version}" for name, version in sorted(selected.items())) + "\n")
    print(f"Locked {len(selected)} runtime distributions")


if __name__ == "__main__":
    main()
