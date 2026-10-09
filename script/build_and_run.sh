#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi
MODE="${1:-run}"
if (( $# > 0 )); then shift; fi
case "$MODE" in
  run|--debug|debug|--logs|logs|--telemetry|telemetry|--verify|verify) ;;
  *) echo "usage: $0 [--dry-run] [run|--debug|--logs|--telemetry|--verify] [--tool studio|review|compare|dataset|train] [-- app arguments]" >&2; exit 2 ;;
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
CONFIGURATION="Release"
if [[ "$MODE" == --debug || "$MODE" == debug ]]; then CONFIGURATION="Debug"; fi
PARENT_APP="$DERIVED_DATA/Build/Products/$CONFIGURATION/Texture Studio.app"
APP_BUNDLE="$PARENT_APP"
if [[ "$TOOL_ROLE" != studio ]]; then APP_BUNDLE="$PARENT_APP/Contents/Applications/$APP_NAME.app"; fi
APP_BINARY="$APP_BUNDLE/Contents/MacOS/$APP_NAME"
if [[ "$DRY_RUN" == true ]]; then
  printf 'configuration=%s\nrole=%s\nparent=%s\napplication=%s\n' "$CONFIGURATION" "$TOOL_ROLE" "$PARENT_APP" "$APP_BUNDLE"
  exit 0
fi

"$ROOT_DIR/script/check_toolchain.sh"
# Stop only this checkout/configuration/role when explicitly rebuilding and relaunching it.
# A running installed Release app or another material tool keeps its process.
while read -r PID EXECUTABLE; do
  if [[ "$EXECUTABLE" == "$APP_BINARY" ]]; then kill "$PID" 2>/dev/null || true; fi
done < <(/bin/ps -axo pid=,comm=)
/usr/bin/xcodebuild -project "$ROOT_DIR/native/TextureStudio/TextureStudio.xcodeproj" \
  -scheme TextureStudio -configuration "$CONFIGURATION" -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$DERIVED_DATA" build
"$ROOT_DIR/script/stage_material_apps.sh" "$PARENT_APP"

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
    /bin/ps -axo comm= | /usr/bin/grep -Fx "$APP_BINARY" >/dev/null
    echo "$APP_NAME launched ($CONFIGURATION): $APP_BUNDLE"
    ;;
esac
