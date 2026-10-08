#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then echo "usage: $0 /path/to/Texture Studio.app" >&2; exit 2; fi
SOURCE_APP="$1"
PRODUCTS="$(dirname "$SOURCE_APP")"
for ROLE in review compare dataset train; do
  case "$ROLE" in
    review) TOOL_NAME="Material Review" ;;
    compare) TOOL_NAME="Checkpoint Compare" ;;
    dataset) TOOL_NAME="Material Dataset" ;;
    train) TOOL_NAME="Material Trainer" ;;
  esac
  STAGING="$(mktemp -d "$PRODUCTS/.material-tool.XXXXXX")"
  trap 'rm -rf "$STAGING"' EXIT
  /usr/bin/ditto "$SOURCE_APP" "$STAGING/$TOOL_NAME.app"
  TOOL_APP="$STAGING/$TOOL_NAME.app"
  mv "$TOOL_APP/Contents/MacOS/Texture Studio" "$TOOL_APP/Contents/MacOS/$TOOL_NAME"
  PLIST="$TOOL_APP/Contents/Info.plist"
  /usr/libexec/PlistBuddy -c "Set :CFBundleExecutable $TOOL_NAME" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleName $TOOL_NAME" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $TOOL_NAME" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier org.ipde.material-$ROLE" "$PLIST"
  /usr/libexec/PlistBuddy -c "Add :MaterialToolRole string $ROLE" "$PLIST"
  /usr/bin/codesign --force --sign - --options runtime "$TOOL_APP" >/dev/null
  /usr/bin/codesign --verify --deep --strict "$TOOL_APP"
  if [[ -d "$PRODUCTS/$TOOL_NAME.app" ]]; then rm -rf "$PRODUCTS/$TOOL_NAME.app"; fi
  mv "$TOOL_APP" "$PRODUCTS/$TOOL_NAME.app"
  rm -rf "$STAGING"
  trap - EXIT
done
