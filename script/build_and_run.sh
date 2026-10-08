#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
if (( $# > 0 )); then shift; fi
case "$MODE" in
  run|--debug|debug|--logs|logs|--telemetry|telemetry|--verify|verify) ;;
  *) echo "usage: $0 [run|--debug|--logs|--telemetry|--verify] [-- app arguments]" >&2; exit 2 ;;
esac
if [[ "${1:-}" == -- ]]; then shift; fi
TOOL_ROLE="studio"
if [[ "${1:-}" == --tool ]]; then
  TOOL_ROLE="${2:-}"
  shift 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Texture Studio"
BUNDLE_ID="org.ipde.texture-studio"
case "$TOOL_ROLE" in
  studio) ;;
  review) APP_NAME="Material Review"; BUNDLE_ID="org.ipde.material-review" ;;
  compare) APP_NAME="Checkpoint Compare"; BUNDLE_ID="org.ipde.material-compare" ;;
  dataset) APP_NAME="Material Dataset"; BUNDLE_ID="org.ipde.material-dataset" ;;
  train) APP_NAME="Material Trainer"; BUNDLE_ID="org.ipde.material-train" ;;
  *) echo "Unknown material tool: $TOOL_ROLE" >&2; exit 2 ;;
esac
DERIVED_DATA="${TEXTURE_STUDIO_DERIVED_DATA:-$ROOT_DIR/build/TextureStudio}"
APP_BUNDLE="$DERIVED_DATA/Build/Products/Debug/$APP_NAME.app"
APP_BINARY="$APP_BUNDLE/Contents/MacOS/$APP_NAME"

"$ROOT_DIR/script/check_toolchain.sh"
/usr/bin/pkill -x "$APP_NAME" >/dev/null 2>&1 || true
/usr/bin/xcodebuild -project "$ROOT_DIR/native/TextureStudio/TextureStudio.xcodeproj" \
  -scheme TextureStudio -configuration Debug -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$DERIVED_DATA" build
"$ROOT_DIR/script/stage_material_apps.sh" "$DERIVED_DATA/Build/Products/Debug/Texture Studio.app"

open_app() {
  if (( $# > 0 )); then /usr/bin/open -n "$APP_BUNDLE" --args "$@"
  else /usr/bin/open -n "$APP_BUNDLE"
  fi
}

case "$MODE" in
  run) open_app "$@" ;;
  --debug|debug) /usr/bin/lldb -- "$APP_BINARY" "$@" ;;
  --logs|logs)
    open_app "$@"
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    open_app "$@"
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"$BUNDLE_ID\""
    ;;
  --verify|verify)
    open_app "$@"
    sleep 1
    /usr/bin/pgrep -x "$APP_NAME" >/dev/null
    echo "$APP_NAME launched: $APP_BUNDLE"
    ;;
esac
