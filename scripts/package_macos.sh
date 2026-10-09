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
if [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIconFile' "$PLIST" 2>/dev/null || true)" != TextureStudio.icns ]] || \
   [[ ! -s "$APP_BUNDLE/Contents/Resources/TextureStudio.icns" ]]; then
  echo "Texture Studio app icon is missing; rebuild before packaging." >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/usr/bin/python3 -B "$SCRIPT_DIR/verify_material_backend.py" "$APP_BUNDLE/Contents/Resources/MaterialBackend"
UNEXPECTED_PAYLOAD="$(/usr/bin/find "$APP_BUNDLE/Contents" \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' -o -name '*.npy' -o -name '*.npz' -o -name '*.exr' -o -name '*.blend' -o -name '*.mlmodelc' -o -name '*.mlpackage' -o -name pyvenv.cfg \) -print -quit)"
if [[ -n "$UNEXPECTED_PAYLOAD" ]]; then
  echo "Application packaging excludes model weights, numeric experiment maps and runtimes: $UNEXPECTED_PAYLOAD" >&2
  exit 1
fi
for ROLE in review compare dataset train; do
  case "$ROLE" in
    review) TOOL_NAME="Material Review"; ICON_NAME=MaterialReview ;;
    compare) TOOL_NAME="Checkpoint Compare"; ICON_NAME=CheckpointCompare ;;
    dataset) TOOL_NAME="Material Dataset"; ICON_NAME=MaterialDataset ;;
    train) TOOL_NAME="Material Trainer"; ICON_NAME=MaterialTrainer ;;
  esac
  TOOL_APP="$APP_BUNDLE/Contents/Applications/$TOOL_NAME.app"
  TOOL_PLIST="$TOOL_APP/Contents/Info.plist"
  if [[ ! -x "$TOOL_APP/Contents/MacOS/$TOOL_NAME" || ! -f "$TOOL_PLIST" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$TOOL_PLIST")" != "org.ipde.material-$ROLE" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :MaterialToolRole' "$TOOL_PLIST")" != "$ROLE" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIconFile' "$TOOL_PLIST" 2>/dev/null || true)" != "$ICON_NAME.icns" ]] || \
     [[ ! -s "$TOOL_APP/Contents/Resources/$ICON_NAME.icns" ]] || \
     [[ "$(/usr/libexec/PlistBuddy -c 'Print :IPDEBuildConfiguration' "$TOOL_PLIST" 2>/dev/null || true)" != Release ]] || \
     [[ -e "$TOOL_APP/Contents/Applications" ]]; then
    echo "Missing or malformed nested material tool: $TOOL_NAME (run make build)." >&2
    exit 1
  fi
  /usr/bin/python3 -B "$SCRIPT_DIR/verify_material_backend.py" "$TOOL_APP/Contents/Resources/MaterialBackend"
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
