#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 /path/to/Texture\ Studio.app /path/to/output" >&2
  exit 2
fi
APP_BUNDLE="$1"
OUTPUT_DIR="$2"
PLIST="$APP_BUNDLE/Contents/Info.plist"
APP_BINARY="$APP_BUNDLE/Contents/MacOS/Texture Studio"
if [[ ! -f "$PLIST" || ! -x "$APP_BINARY" ]]; then
  echo "Build Texture Studio before packaging." >&2
  exit 1
fi
if [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$PLIST")" != org.ipde.texture-studio ]]; then
  echo "Refusing to package an unrelated application." >&2
  exit 1
fi
if [[ "$(/usr/libexec/PlistBuddy -c 'Print :IPDEBuildConfiguration' "$PLIST" 2>/dev/null || true)" != Release ]]; then
  echo "Package requires an optimized Release build (run make package)." >&2
  exit 1
fi
if [[ "$(/usr/bin/lipo -archs "$APP_BINARY")" != arm64 ]]; then
  echo "Texture Studio package must contain the arm64 app." >&2
  exit 1
fi
DA3_RESOURCES="$APP_BUNDLE/Contents/Resources/DA3Backend"
for REQUIRED_RESOURCE in worker.py setup_runtime.py requirements.txt UPSTREAM_LICENSE UPSTREAM_REVISION upstream/depth_anything_3/api.py upstream/depth_anything_3/configs/da3-giant.yaml; do
  if [[ ! -f "$DA3_RESOURCES/$REQUIRED_RESOURCE" ]]; then
    echo "Missing required standalone DA3 backend resource: $REQUIRED_RESOURCE" >&2
    exit 1
  fi
done
UNEXPECTED_RESOURCE="$(/usr/bin/find "$DA3_RESOURCES" \( -name __pycache__ -o -name '*.pyc' -o -name pyvenv.cfg -o -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' \) -print -quit)"
if [[ -n "$UNEXPECTED_RESOURCE" ]]; then
  echo "DA3 app resources must contain source and dependency pins only: $UNEXPECTED_RESOURCE" >&2
  exit 1
fi
UNEXPECTED_PAYLOAD="$(/usr/bin/find "$APP_BUNDLE/Contents" \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' -o -name '*.npy' -o -name '*.npz' -o -name '*.exr' -o -name '*.blend' -o -name '*.mlmodelc' -o -name '*.mlpackage' -o -name pyvenv.cfg \) -print -quit)"
if [[ -n "$UNEXPECTED_PAYLOAD" ]]; then
  echo "Application packaging excludes model weights, numeric experiment maps and runtimes: $UNEXPECTED_PAYLOAD" >&2
  exit 1
fi
for ROLE in review compare dataset train; do
  case "$ROLE" in
    review) TOOL_NAME="Material Review" ;;
    compare) TOOL_NAME="Checkpoint Compare" ;;
    dataset) TOOL_NAME="Material Dataset" ;;
    train) TOOL_NAME="Material Trainer" ;;
  esac
  TOOL_APP="$APP_BUNDLE/Contents/Applications/$TOOL_NAME.app"
  TOOL_PLIST="$TOOL_APP/Contents/Info.plist"
  if [[ ! -x "$TOOL_APP/Contents/MacOS/$TOOL_NAME" || ! -f "$TOOL_PLIST" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$TOOL_PLIST")" != "org.ipde.material-$ROLE" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :MaterialToolRole' "$TOOL_PLIST")" != "$ROLE" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :IPDEBuildConfiguration' "$TOOL_PLIST" 2>/dev/null || true)" != Release ]] || \
     [[ -e "$TOOL_APP/Contents/Applications" ]]; then
    echo "Missing or malformed nested material tool: $TOOL_NAME (run make build)." >&2
    exit 1
  fi
  /usr/bin/codesign --verify --deep --strict "$TOOL_APP"
done
/usr/bin/codesign --verify --deep --strict "$APP_BUNDLE"
mkdir -p "$OUTPUT_DIR"
PACKAGE_DIR="$(/usr/bin/mktemp -d "$OUTPUT_DIR/.texture-studio-package.XXXXXX")"
trap 'rm -rf "$PACKAGE_DIR"' EXIT
/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE" "$PACKAGE_DIR/Texture-Studio-macos-arm64.zip"
(cd "$PACKAGE_DIR" && /usr/bin/shasum -a 256 Texture-Studio-macos-arm64.zip > Texture-Studio-macos-arm64.sha256)
mv "$PACKAGE_DIR/Texture-Studio-macos-arm64.zip" "$OUTPUT_DIR/Texture-Studio-macos-arm64.zip"
mv "$PACKAGE_DIR/Texture-Studio-macos-arm64.sha256" "$OUTPUT_DIR/Texture-Studio-macos-arm64.sha256"
echo "Packaged Texture Studio: $OUTPUT_DIR/Texture-Studio-macos-arm64.zip"
