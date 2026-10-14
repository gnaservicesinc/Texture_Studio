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
/usr/bin/xcrun swift - "$bundle" "$identifier" "$executable" <<'SWIFT'
import Foundation
let arguments = Array(CommandLine.arguments.dropFirst())
let bundle = URL(fileURLWithPath: arguments[0], isDirectory: true)
let info = ["CFBundleIdentifier": arguments[1], "CFBundleExecutable": arguments[2],
            "CFBundleName": arguments[2], "CFBundlePackageType": "APPL", "IPDEBuildConfiguration": "Release"]
try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
    .write(to: bundle.appendingPathComponent("Contents/Info.plist"))
SWIFT
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
/usr/bin/xcrun swift - "$SUITE_TEST_LOG" "$STAGING" <<'SWIFT'
import Foundation
let arguments = Array(CommandLine.arguments.dropFirst())
let events = try String(contentsOfFile: arguments[0], encoding: .utf8).split(whereSeparator: \.isNewline).map(String.init)
precondition(events == ["build:Texture Studio", "build:Loom Factory", "build:Loom Workshop",
                       "install:Loom Factory", "install:Loom Workshop"], "Unexpected build/install order: \(events)")
let applications = URL(fileURLWithPath: arguments[1], isDirectory: true).appendingPathComponent("Applications")
for name in ["Texture Studio", "Loom Factory", "Loom Workshop"] {
    let verify = Process(); verify.executableURL = URL(fileURLWithPath: "/usr/bin/codesign")
    verify.arguments = ["--verify", "--deep", "--strict", applications.appendingPathComponent(name + ".app").path]
    try verify.run(); verify.waitUntilExit()
    precondition(verify.terminationStatus == 0, "Signature verification failed: \(name)")
}
precondition(!FileManager.default.fileExists(atPath: applications.appendingPathComponent("Texture Studio.app/marker").path),
             "Previous Texture Studio build was not replaced")
SWIFT

# Dry-run verifies real-install close/launch order and automatic recovery of
# missing separate checkouts without changing the checkout or user applications.
"$FIXTURE/script/install_suite.sh" --dry-run --factory-repo "$FIXTURE/Missing Factory" --workshop-repo "$FIXTURE/Missing Workshop" > "$SCRATCH/dry-run.log"
[[ ! -e "$FIXTURE/Missing Factory" && ! -e "$FIXTURE/Missing Workshop" ]]
"$FIXTURE/script/install_suite.sh" --dry-run --destdir "$STAGING" > "$SCRATCH/staging-dry-run.log"
"$FIXTURE/script/install_suite.sh" --dry-run --no-launch > "$SCRATCH/no-launch-dry-run.log"
ln -s / "$SCRATCH/root-alias"
"$FIXTURE/script/install_suite.sh" --dry-run --destdir "$SCRATCH/root-alias" > "$SCRATCH/aliased-root-dry-run.log"
/usr/bin/xcrun swift - "$SCRATCH" <<'SWIFT'
import Foundation
let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
enum Quote { case none, single, double }
func shellWords(_ line: String) -> [String] {
    var words: [String] = [], word = "", quote = Quote.none, escaped = false, started = false
    for character in line {
        if escaped {
            if quote == .double && character != "\"" && character != "\\" { word.append("\\") }
            word.append(character); escaped = false; started = true
        } else if quote == .single {
            if character == "'" { quote = .none } else { word.append(character) }
        } else if character == "\\" {
            escaped = true; started = true
        } else if quote == .double {
            if character == "\"" { quote = .none } else { word.append(character) }
        } else if character == "'" {
            quote = .single; started = true
        } else if character == "\"" {
            quote = .double; started = true
        } else if character.isWhitespace {
            if started { words.append(word); word = ""; started = false }
        } else {
            word.append(character); started = true
        }
    }
    precondition(!escaped && quote == .none, "Invalid shell quoting: \(line)")
    if started { words.append(word) }
    return words
}
func commands(_ name: String) throws -> [[String]] {
    try String(contentsOf: root.appendingPathComponent(name), encoding: .utf8)
        .split(whereSeparator: \.isNewline).filter { $0.hasPrefix("/") }.map { shellWords(String($0)) }
}
let real = try commands("dry-run.log")
let clones = real.filter { Array($0.prefix(2)) == ["/usr/bin/git", "clone"] }
precondition(clones.map { $0[2] } == ["https://github.com/gnaservicesinc/LoomFactory.git",
                                    "https://github.com/gnaservicesinc/LoomWorkshop.git"], "Missing repository recovery commands")
let builds = real.indices.filter { real[$0][0].hasSuffix("build-app.sh") || real[$0][0] == "/usr/bin/make" }
precondition(builds.count == 3, "Expected three application builds")
precondition(real[builds[0]].contains("CONFIGURATION=Release"), "Suite install must build Release")
let close = real.firstIndex { $0.last == "close" }!
let launch = real.firstIndex { $0.last == "launch" }!
let installs = real.indices.filter { real[$0].contains { $0.contains("install_macos.swift") || $0.hasSuffix("install-app.sh") } }
precondition(installs.count == 3 && builds.max()! < close && close < installs.min()! &&
             installs.min()! <= installs.max()! && installs.max()! < launch, "Invalid build/close/install/launch order")
let staged = try commands("staging-dry-run.log")
precondition(!staged.joined().contains { $0.contains("suite_app_control.swift") }, "Staging must not control running applications")
let noLaunch = try commands("no-launch-dry-run.log")
precondition(noLaunch.contains { $0.last == "close" }, "Real installation must close applications")
precondition(!noLaunch.contains { $0.last == "launch" }, "--no-launch must not launch applications")
let aliasedRoot = try commands("aliased-root-dry-run.log")
precondition(aliasedRoot.contains { $0.last == "close" }, "Symlink to / must close applications")
precondition(aliasedRoot.contains { $0.last == "launch" }, "Symlink to / must launch applications")
SWIFT
/usr/bin/xcrun swift "$ROOT_DIR/script/suite_app_control.swift" --self-test
printf 'Three-application install checks passed\n'
