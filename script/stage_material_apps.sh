#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then echo "usage: $0 /path/to/Texture Studio.app" >&2; exit 2; fi
SOURCE_APP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
PRODUCTS="$(dirname "$SOURCE_APP")"
if [[ ! -x "$SOURCE_APP/Contents/MacOS/Texture Studio" ]] || \
   [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$SOURCE_APP/Contents/Info.plist")" != org.ipde.texture-studio ]]; then
  echo "Stage tools only from a complete Texture Studio application." >&2; exit 1
fi
STAGING="$(mktemp -d "$PRODUCTS/.material-suite.XXXXXX")"
BACKUP="$STAGING/previous.app"
cleanup() {
  if [[ -d "$BACKUP" && ! -e "$SOURCE_APP" ]]; then
    mv "$BACKUP" "$SOURCE_APP" || { echo "Previous application retained at $BACKUP" >&2; return; }
  fi
  rm -rf "$STAGING"
}
trap cleanup EXIT
PARENT_APP="$STAGING/Texture Studio.app"
/usr/bin/ditto "$SOURCE_APP" "$PARENT_APP"
# Never copy previously embedded tools into each new child.
rm -rf "$PARENT_APP/Contents/Applications"
mkdir "$STAGING/Applications"
SIGN_IDENTITY="${TEXTURE_STUDIO_SIGN_IDENTITY:--}"
for ROLE in review compare dataset train; do
  case "$ROLE" in
    review) TOOL_NAME="Material Review"; ICON_NAME=MaterialReview ;;
    compare) TOOL_NAME="Checkpoint Compare"; ICON_NAME=CheckpointCompare ;;
    dataset) TOOL_NAME="Material Dataset"; ICON_NAME=MaterialDataset ;;
    train) TOOL_NAME="Material Trainer"; ICON_NAME=MaterialTrainer ;;
  esac
  TOOL_APP="$STAGING/Applications/$TOOL_NAME.app"
  /usr/bin/ditto "$PARENT_APP" "$TOOL_APP"
  mv "$TOOL_APP/Contents/MacOS/Texture Studio" "$TOOL_APP/Contents/MacOS/$TOOL_NAME"
  PLIST="$TOOL_APP/Contents/Info.plist"
  /usr/libexec/PlistBuddy -c "Set :CFBundleExecutable $TOOL_NAME" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleName $TOOL_NAME" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $TOOL_NAME" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier org.ipde.material-$ROLE" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile $ICON_NAME.icns" "$PLIST"
  if [[ ! -s "$TOOL_APP/Contents/Resources/$ICON_NAME.icns" ]]; then
    echo "Missing $TOOL_NAME icon; rebuild Texture Studio before staging tools." >&2; exit 1
  fi
  # The parent keeps all role artwork for restaging. Each child needs only its
  # own icon, avoiding twenty redundant full-resolution copies in the suite.
  for OTHER_ICON in TextureStudio MaterialReview CheckpointCompare MaterialDataset MaterialTrainer; do
    if [[ "$OTHER_ICON" != "$ICON_NAME" ]]; then
      rm -f "$TOOL_APP/Contents/Resources/$OTHER_ICON.icns"
    fi
  done
  /usr/libexec/PlistBuddy -c "Delete :MaterialToolRole" "$PLIST" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Add :MaterialToolRole string $ROLE" "$PLIST"
  /usr/bin/codesign --force --sign "$SIGN_IDENTITY" --preserve-metadata=entitlements,flags,runtime "$TOOL_APP" >/dev/null
  /usr/bin/codesign --verify --deep --strict "$TOOL_APP"
done
mv "$STAGING/Applications" "$PARENT_APP/Contents/Applications"
# Seal the parent only after all nested app signatures are complete.
/usr/bin/codesign --force --sign "$SIGN_IDENTITY" --preserve-metadata=entitlements,flags,runtime "$PARENT_APP" >/dev/null
/usr/bin/codesign --verify --deep --strict "$PARENT_APP"
mv "$SOURCE_APP" "$BACKUP"
mv "$PARENT_APP" "$SOURCE_APP"
rm -rf "$BACKUP"
echo "Bundled four material tools inside $SOURCE_APP/Contents/Applications"
