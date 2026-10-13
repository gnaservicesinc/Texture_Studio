#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/loom-suite-install-tests.XXXXXX")"
trap '/bin/rm -rf "$SCRATCH"' EXIT
FIXTURE="$SCRATCH/checkout with spaces"
STAGING="$SCRATCH/staging with spaces"
mkdir -p "$FIXTURE/script" "$FIXTURE/LoomFactory/Scripts" "$FIXTURE/LoomWorkshop/Scripts"
cp "$ROOT_DIR/script/install_suite.sh" "$ROOT_DIR/script/install_macos.swift" "$FIXTURE/script/"
export SUITE_TEST_LOG="$SCRATCH/events"
cat > "$FIXTURE/script/fixture_build.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
name="$1" identifier="$2" executable="$3" bundle="$4"
printf 'build:%s\n' "$name" >> "$SUITE_TEST_LOG"
if [[ "${SUITE_TEST_FAIL:-}" == "$name" ]]; then exit 7; fi
mkdir -p "$bundle/Contents/MacOS"
cp /usr/bin/true "$bundle/Contents/MacOS/$executable"
python3 - "$bundle" "$identifier" "$executable" <<'PY'
import pathlib, plistlib, sys
bundle, identifier, executable = sys.argv[1:]
(pathlib.Path(bundle) / 'Contents/Info.plist').write_bytes(plistlib.dumps({
    'CFBundleIdentifier': identifier, 'CFBundleExecutable': executable,
    'CFBundleName': executable, 'CFBundlePackageType': 'APPL',
    'IPDEBuildConfiguration': 'Release',
}))
PY
/usr/bin/codesign --force --sign - "$bundle"
SH
cat > "$FIXTURE/Makefile" <<'MAKE'
.PHONY: build
build:
	@test "$(CONFIGURATION)" = Release
	@./script/fixture_build.sh "Texture Studio" org.ipde.texture-studio "Texture Studio" "$(DERIVED_DATA)/Build/Products/Release/Texture Studio.app"
MAKE
for product in Factory Workshop; do
    directory="$FIXTURE/Loom$product"
    cat > "$directory/Scripts/build-app.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
root="\$(cd "\$(dirname "\$0")/.." && pwd)"
"\$root/../script/fixture_build.sh" "Loom $product" com.gnaservicesinc.Loom$product Loom$product "\$root/.build/application/Loom $product.app"
SH
    cat > "$directory/Scripts/install-app.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
[[ "\$1" == --skip-build && "\$2" == --destdir && "\$#" == 3 ]]
root="\$(cd "\$(dirname "\$0")/.." && pwd)"
printf 'install:Loom $product\n' >> "\$SUITE_TEST_LOG"
mkdir -p "\$3/Applications"
/usr/bin/ditto "\$root/.build/application/Loom $product.app" "\$3/Applications/Loom $product.app"
SH
    chmod +x "$directory/Scripts/build-app.sh" "$directory/Scripts/install-app.sh"
done
chmod +x "$FIXTURE/script/fixture_build.sh" "$FIXTURE/script/install_suite.sh"

# A failed later build must leave every installed application untouched.
mkdir -p "$STAGING/Applications/Texture Studio.app"
printf 'previous build\n' > "$STAGING/Applications/Texture Studio.app/marker"
if SUITE_TEST_FAIL='Loom Factory' "$FIXTURE/script/install_suite.sh" --destdir "$STAGING" > "$SCRATCH/failed-build.log" 2>&1; then
    printf 'A failed Factory build was unexpectedly accepted.\n' >&2; exit 1
fi
[[ "$(cat "$STAGING/Applications/Texture Studio.app/marker")" == 'previous build' ]]
[[ ! -e "$STAGING/Applications/Loom Factory.app" && ! -e "$STAGING/Applications/Loom Workshop.app" ]]
: > "$SUITE_TEST_LOG"
CONFIGURATION=Debug "$FIXTURE/script/install_suite.sh" --destdir "$STAGING" > "$SCRATCH/success.log" 2>&1
python3 - "$SUITE_TEST_LOG" "$STAGING" <<'PY'
import pathlib, subprocess, sys
events = pathlib.Path(sys.argv[1]).read_text().splitlines()
assert events == ['build:Texture Studio', 'build:Loom Factory', 'build:Loom Workshop',
                  'install:Loom Factory', 'install:Loom Workshop'], events
for name in ['Texture Studio', 'Loom Factory', 'Loom Workshop']:
    subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict',
                    str(pathlib.Path(sys.argv[2]) / 'Applications' / (name + '.app'))], check=True)
assert not (pathlib.Path(sys.argv[2]) / 'Applications/Texture Studio.app/marker').exists()
PY

# Dry-run verifies real-install close/launch order and automatic recovery of
# missing separate checkouts without changing the checkout or user applications.
"$FIXTURE/script/install_suite.sh" --dry-run --factory-repo "$FIXTURE/Missing Factory" --workshop-repo "$FIXTURE/Missing Workshop" > "$SCRATCH/dry-run.log"
[[ ! -e "$FIXTURE/Missing Factory" && ! -e "$FIXTURE/Missing Workshop" ]]
"$FIXTURE/script/install_suite.sh" --dry-run --destdir "$STAGING" > "$SCRATCH/staging-dry-run.log"
"$FIXTURE/script/install_suite.sh" --dry-run --no-launch > "$SCRATCH/no-launch-dry-run.log"
ln -s / "$SCRATCH/root-alias"
"$FIXTURE/script/install_suite.sh" --dry-run --destdir "$SCRATCH/root-alias" > "$SCRATCH/aliased-root-dry-run.log"
python3 - "$SCRATCH" <<'PY'
import pathlib, shlex, sys
root = pathlib.Path(sys.argv[1])
def commands(name):
    return [shlex.split(line) for line in (root / name).read_text().splitlines()
            if line.startswith('/')]
real = commands('dry-run.log')
clones = [line for line in real if line[:2] == ['/usr/bin/git', 'clone']]
assert [line[2] for line in clones] == [
    'https://github.com/gnaservicesinc/LoomFactory.git',
    'https://github.com/gnaservicesinc/LoomWorkshop.git']
builds = [i for i, line in enumerate(real) if line[0].endswith('build-app.sh') or line[0] == '/usr/bin/make']
assert len(builds) == 3
assert 'CONFIGURATION=Release' in real[builds[0]]
close = next(i for i, line in enumerate(real) if line[-1] == 'close')
launch = next(i for i, line in enumerate(real) if line[-1] == 'launch')
installs = [i for i, line in enumerate(real) if any('install_macos.swift' in arg or arg.endswith('install-app.sh') for arg in line)]
assert len(installs) == 3 and max(builds) < close < min(installs) <= max(installs) < launch
staged = commands('staging-dry-run.log')
assert not any('suite_app_control.swift' in arg for line in staged for arg in line)
no_launch = commands('no-launch-dry-run.log')
assert any(line[-1] == 'close' for line in no_launch)
assert not any(line[-1] == 'launch' for line in no_launch)
aliased_root = commands('aliased-root-dry-run.log')
assert any(line[-1] == 'close' for line in aliased_root)
assert any(line[-1] == 'launch' for line in aliased_root)
PY
/usr/bin/xcrun swift "$ROOT_DIR/script/suite_app_control.swift" --self-test
printf 'Three-application install checks passed\n'
