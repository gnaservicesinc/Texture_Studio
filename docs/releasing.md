# Versioning and macOS releases

IPDE Studio starts this development line at **0.9.0**. The 0.9.x line is
pre-release software. Until 1.0.0, projects, save files, datasets, checkpoints
and settings can change incompatibly. Preserve original photographs,
scientific exports and useful checkpoints before upgrading. Migration support
is not a compatibility promise for this development line.

The version must agree in `CMakeLists.txt`, `pyproject.toml` and
`src/ipde/__init__.py`. `scripts/check_release_version.py` verifies them without
loading optional inference libraries. Release tags use `v0.9.0`, `v0.9.1`, and
so on. No release tag is created by building locally.

## Local packaging and installation

Install the requirements into a Python environment, then run:

```sh
make build
make package
make install
# For a staging directory instead of the system Applications folder:
make DESTDIR=/path/to/staging/dir install
```

`make build` embeds the Qt frameworks and plugins in every native application,
including the nested applications in IPDE Studio. `make package` creates
`build/dist/IPDE Studio.app`, adding a single shared Python runtime and the
runtime dependency closure pinned in `requirements-release-macos.txt`.
It requires the versions in that file; unrelated installed packages are omitted.
Its dependency audit rejects
references to an external Qt, Python, Homebrew, or development library. Native
code is ad-hoc signed and verified after deployment. The recursive audit also
reads each native binary's required macOS version and rejects one newer than
the version declared by the app. The app is not notarized.

`make setup` installs the base, teacher and optional dataset dependencies with
the release constraints. To upgrade these intentionally, install the chosen
versions, run `scripts/lock-release-dependencies.py`, then rerun validation.
The lock is the dependency closure of those features for macOS Python 3.13
and 3.14; it includes required extras and package license metadata.
Teacher sources, model weights, datasets and personal workspace files are
user-managed and are not included. Configure model sources/weights on the
destination Mac before using model-based operations.

`make install` copies the finished app into `/Applications/IPDE Studio.app`.
`DESTDIR` is a staging **root**, so the example above produces
`/path/to/staging/dir/Applications/IPDE Studio.app`. Installation refuses to
replace a running installed app or one of its subapps. It copies and verifies a
temporary replacement before removing the old app. System Applications may
require an account with permission to write there.

`PYTHON`, `PYTHON_BASE`, `QT_CMAKE`, `BUILD_DIR`, `BUILD_TYPE`, and `CMAKE_ARGS`
can be overridden for another development installation. Qt is otherwise
discovered from installed `/opt/Qt/6.*/macos` versions.
The release workflow pins Qt 6.11.1. A newer Qt may require a newer macOS
version: [Qt 6.12 requires macOS 14.4](https://doc.qt.io/qt-6.12/macos.html). When building with that version, set
`CMAKE_ARGS=-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4` so the app declares its actual
minimum. The audit deliberately rejects a bundle claiming an older minimum
than one of its libraries supports.

## GitHub Actions

`.github/workflows/macos-release.yml` builds the Apple Silicon package,
runs native/backend verification, deploys and audits the runtime, verifies
signatures, and saves ZIP artifacts and SHA-256 checksums. Pushes to `main`,
`release/**`, pull requests and manual runs build artifacts. A matching `v*`
tag additionally publishes a GitHub Release after the package checks succeed.
All versions below 1.0.0 are marked as pre-releases. Building or pushing a
branch does not publish a release.

The initial release supports **Apple Silicon Macs with macOS 14 or later**.
The application and Swift helper use that deployment floor; the current
PyTorch runtime also requires macOS 14. Intel packages are not
produced because current supported PyTorch wheels are required for this
training application. See the [PyTorch macOS x86 support announcement](https://docs.pytorch.org/blog/pytorch2-2/).

The release uses the [GitHub macOS runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and Qt's [macOS deployment tool](https://doc.qt.io/qt-6/macos-deployment.html).
It does not contain signing credentials or a notarization submission. A
downloaded app may need first-launch approval in macOS Privacy & Security.

Before tagging, review the changes, run the checks and confirm package contents.
Update the version consistently and create an annotated tag at the reviewed
commit. Push that tag explicitly only when publishing is intended. Inspect the
completed workflow and release assets; a pushed tag alone does not establish
that a downloadable release exists.

## Stable release branches from 1.0.0

At the first stable release, create `release/1.0` at the `v1.0.0` commit. Each
supported stable minor line gets its own branch: `release/1.1`, `release/1.2`,
and so on. `main` continues forward development. Create patch releases from
the appropriate supported branch, for example `v1.0.1` from `release/1.0`.

Backport selected fixes with their relevant regression coverage. Keep new
features and format-breaking changes on `main`. Document supported release
lines and any file-format migration guarantees when 1.0.0 is reached. Before
publishing a backport, validate that branch's package rather than relying on
validation of the original fix on `main`.
