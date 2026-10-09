#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/texture-build-tests.XXXXXX")"
trap '/bin/rm -rf "$SCRATCH"' EXIT
FIXTURE="$SCRATCH/repository"
APPLICATION="$SCRATCH/Texture Studio.app"
mkdir -p "$APPLICATION/Contents/Resources"
"$SCRIPT_DIR/verify_native_bundle.sh" "$APPLICATION"
for RESOURCE in worker.py worker.pyc worker.pyo libpython3.14.dylib pyvenv.cfg; do
  printf 'retired resource fixture\n' > "$APPLICATION/Contents/Resources/$RESOURCE"
  if "$SCRIPT_DIR/verify_native_bundle.sh" "$APPLICATION" > "$SCRATCH/error" 2>&1; then
    echo "Retired resource unexpectedly accepted: $RESOURCE" >&2; exit 1
  fi
  /bin/rm "$APPLICATION/Contents/Resources/$RESOURCE"
done
for RESOURCE in .venv venv __pycache__ ipde.egg-info Python.framework MaterialBackend; do
  mkdir "$APPLICATION/Contents/Resources/$RESOURCE"
  if "$SCRIPT_DIR/verify_native_bundle.sh" "$APPLICATION" > "$SCRATCH/error" 2>&1; then
    echo "Retired resource directory unexpectedly accepted: $RESOURCE" >&2; exit 1
  fi
  rmdir "$APPLICATION/Contents/Resources/$RESOURCE"
done
"$SCRIPT_DIR/verify_native_bundle.sh" "$APPLICATION"
/usr/bin/xcrun swift "$SCRIPT_DIR/install_macos.swift" --self-test
/usr/bin/xcrun swift "$SCRIPT_DIR/check_release_version.swift" --root "$REPOSITORY"
if TAG=v0.0.0 REF_TYPE=tag /usr/bin/xcrun swift "$SCRIPT_DIR/check_release_version.swift" --root "$REPOSITORY" > "$SCRATCH/error" 2>&1; then
  echo "Mismatched release tag unexpectedly accepted" >&2; exit 1
fi
# Check mismatched native metadata independently of the actual checkout.
mkdir -p "$FIXTURE/src/TextureStudio/TextureStudio.xcodeproj"
/bin/cp "$REPOSITORY/VERSION" "$FIXTURE/VERSION"
printf 'MARKETING_VERSION = 100.0.0;\n' > "$FIXTURE/src/TextureStudio/TextureStudio.xcodeproj/project.pbxproj"
if /usr/bin/xcrun swift "$SCRIPT_DIR/check_release_version.swift" --root "$FIXTURE" > "$SCRATCH/error" 2>&1; then
  echo "Mismatched release sources unexpectedly accepted" >&2; exit 1
fi
echo "Native build tool checks passed"
