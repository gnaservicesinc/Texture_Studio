# Building and releasing Texture Studio

Texture Studio is the standalone native SwiftUI application in
`native/TextureStudio/TextureStudio.xcodeproj`. The shared scheme is
`TextureStudio`; its executable and application bundle are named **Texture
Studio**. The bundle identifier is `org.ipde.texture-studio`.

The app requires an Apple Silicon Mac, macOS 26 or later, and a full Xcode
installation containing the macOS 26 SDK or later. Command Line Tools alone are
insufficient. The project uses Apple's SwiftUI, AppKit, ImageIO, Core Image,
Metal, and Core ML frameworks. Qt, CMake, Ninja, Linux, and Windows app build
paths have been retired. The Python extraction and research tools remain
separate from the app. The app bundles its current Python material backend
source and license under `Contents/Resources/MaterialBackend`. Python,
PyTorch/MPS, and model binaries remain outside the app bundle; configure their
paths in the material-model workspace.

## Local development

Open the project in Xcode and select the TextureStudio scheme, or run:

```sh
make build
./script/build_and_run.sh
```

`make build` builds Release into `build/TextureStudio/Build/Products/Release`.
The run script builds and launches Release by default. Only `--debug` chooses
Debug and LLDB. It stops only the matching app in this checkout/configuration;
an installed app is left running. Its optional modes are `--debug`,
`--logs`, `--telemetry`, and `--verify`. Extra app arguments follow `--`.
The Codex **Run Release** action uses this script through
`.codex/environments/environment.toml`.

```sh
./script/build_and_run.sh --verify
./script/build_and_run.sh run -- --open /path/to/surface.heic
make test-native
make smoke
```

The native XCTest target tests geometry, map exports, and model management.
The Core Image Metal kernels are compiled with `metal -fcikernel` and
`metallib -cikernel` by the Xcode build and embedded in the app resources.
No model downloads are needed for the deterministic native checks. Real material
inference checks require the separately installed selected model and Metal runtime.

Python backend checks are optional for app-only development. `make setup`
installs the test requirements; `make test-python` uses pytest to run both
function-based regressions and the unittest cases:

```sh
make setup
make test-python
```

`make test` runs both suites. `CONFIGURATION=Debug` changes the build
configuration; `DERIVED_DATA=/path/to/build` changes the Makefile output root.
The run script accepts `TEXTURE_STUDIO_DERIVED_DATA` for its output root.

## Packages and versions

```sh
make package
make release-check
# Optional: install the built app, after closing Texture Studio.
make install
# Or install under a staging root:
make DESTDIR=/path/to/staging install
```

Packaging verifies the app's signature, identity, and arm64 executable, then
creates `dist/Texture-Studio-macos-arm64.zip` and its SHA-256 checksum. The app
contains native resources, the material Python backend, and the four signed
material tools under `Contents/Applications`. Packaging checks the backend's
complete source list and license in the parent and each child using the same
list as the Xcode staging step. Children are signed before
the parent bundle is sealed. The installer refuses Debug builds and refuses to
replace a running installed parent or child; close them and rerun `make install`.
Optional models are stored outside the application bundle.

Primary development commits stay on `main`. Commit and push source changes after
validation, build/package Release, and install in `/Applications` when the installed
suite is closed. Model weights, datasets and generated outputs are excluded;
configure Git LFS before embedding model binaries. Tags identify deliberately
published versions; building a Release configuration does not publish a GitHub Release.

The version must agree in `pyproject.toml`, `src/ipde/__init__.py`, and every
`MARKETING_VERSION` entry in the Xcode project. The tag must be `v` followed by
that version. `scripts/check_release_version.py` checks these without importing
model runtimes. The current 0.9.x series remains a development prerelease.

The macOS GitHub workflow tests the native app and retained Python tools on
`macos-26`, builds the package, and publishes tagged artifacts. A workflow
change is not evidence that remote CI has passed. A local package is not a
published GitHub Release.

Local and CI packages use ad-hoc signing and are not notarized. A public trusted
release additionally requires a Developer ID identity, hardened runtime,
notarization, and stapling; credentials are never stored in this repository.
Use Xcode's archive/distribution workflow when preparing that release.
