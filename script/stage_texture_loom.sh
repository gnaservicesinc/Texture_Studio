#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then echo 'usage: stage_texture_loom.sh /path/to/Texture Studio.app' >&2; exit 2; fi
APP_BUNDLE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/usr/bin/xcrun swift "$SCRIPT_DIR/stage_texture_loom.swift" "$APP_BUNDLE" "${TEXTURE_LOOM_PACKAGE:-}"
/usr/bin/xcrun swift "$SCRIPT_DIR/stage_photo_enhancers.swift" "$APP_BUNDLE"
/usr/bin/codesign --force --sign - --options runtime "$APP_BUNDLE"
/usr/bin/codesign --verify --deep --strict "$APP_BUNDLE"
