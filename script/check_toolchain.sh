#!/usr/bin/env bash
set -euo pipefail

if [[ "$(/usr/bin/uname -s)" != Darwin || "$(/usr/bin/uname -m)" != arm64 ]]; then
  echo "Texture Studio requires an Apple Silicon Mac with macOS 26 or later." >&2
  exit 1
fi
DEVELOPER_PATH="$(/usr/bin/xcode-select -p)"
if [[ "$DEVELOPER_PATH" != *.app/Contents/Developer || ! -x "$DEVELOPER_PATH/usr/bin/xcodebuild" ]]; then
  echo "Select a full Xcode installation (Xcode 26 or later), then run xcodebuild -runFirstLaunch." >&2
  echo "The standalone Command Line Tools do not include the app build toolchain." >&2
  exit 1
fi
SDK_VERSION="$(/usr/bin/xcrun --sdk macosx --show-sdk-version)"
if (( ${SDK_VERSION%%.*} < 26 )); then
  echo "Texture Studio requires the macOS 26 SDK or later; selected SDK: $SDK_VERSION." >&2
  exit 1
fi
OS_VERSION="$(/usr/bin/sw_vers -productVersion)"
if (( ${OS_VERSION%%.*} < 26 )); then
  echo "Texture Studio requires macOS 26 or later; installed version: $OS_VERSION." >&2
  exit 1
fi
