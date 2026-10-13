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
"$SCRIPT_DIR/../script/verify_native_bundle.sh" "$APP_BUNDLE"
"$SCRIPT_DIR/../script/verify_texture_loom.sh" "$APP_BUNDLE"
/usr/bin/codesign --verify --deep --strict "$APP_BUNDLE"
mkdir -p "$OUTPUT_DIR"
PACKAGE_DIR="$(/usr/bin/mktemp -d "$OUTPUT_DIR/.texture-studio-package.XXXXXX")"
trap 'rm -rf "$PACKAGE_DIR"' EXIT
/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE" "$PACKAGE_DIR/Texture-Studio-macos-arm64.zip"
(cd "$PACKAGE_DIR" && /usr/bin/shasum -a 256 Texture-Studio-macos-arm64.zip > Texture-Studio-macos-arm64.sha256)
mv "$PACKAGE_DIR/Texture-Studio-macos-arm64.zip" "$OUTPUT_DIR/Texture-Studio-macos-arm64.zip"
mv "$PACKAGE_DIR/Texture-Studio-macos-arm64.sha256" "$OUTPUT_DIR/Texture-Studio-macos-arm64.sha256"
echo "Packaged Texture Studio: $OUTPUT_DIR/Texture-Studio-macos-arm64.zip"
