#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DERIVED_DATA="$ROOT_DIR/build/TextureStudio"
DESTDIR=/
FACTORY_REPO="${LOOM_FACTORY_REPO:-$ROOT_DIR/LoomFactory}"
WORKSHOP_REPO="${LOOM_WORKSHOP_REPO:-$ROOT_DIR/LoomWorkshop}"
LAUNCH=true
DRY_RUN=false

while (( $# )); do
    case "$1" in
        --derived-data|--destdir|--factory-repo|--workshop-repo)
            if (( $# < 2 )); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
            case "$1" in
                --derived-data) DERIVED_DATA="$2" ;;
                --destdir) DESTDIR="$2" ;;
                --factory-repo) FACTORY_REPO="$2" ;;
                --workshop-repo) WORKSHOP_REPO="$2" ;;
            esac
            shift 2
            ;;
        --no-launch) LAUNCH=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help)
            printf 'usage: %s [--destdir ROOT] [--derived-data PATH] [--factory-repo PATH] [--workshop-repo PATH] [--no-launch] [--dry-run]\n' "$0"
            exit 0
            ;;
        *) printf 'Unknown suite install option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Make all paths absolute before invoking independent repositories.
for variable in DERIVED_DATA DESTDIR FACTORY_REPO WORKSHOP_REPO; do
    value="${!variable}"
    if [[ "$value" != /* ]]; then printf -v "$variable" '%s/%s' "$ROOT_DIR" "$value"; fi
done
# Resolve aliases before deciding whether this is a real installation. A symlink
# to / must not bypass the running-app protection as an alleged staging tree.
DESTDIR="$(/usr/bin/xcrun swift -e 'import Foundation; print(URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL.resolvingSymlinksInPath().path)' "$DESTDIR")"
# Staged installs neither quit a user's apps nor open staged copies.
if [[ "$DESTDIR" != / ]]; then LAUNCH=false; fi

run() {
    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

ensure_repository() {
    local directory="$1" remote="$2"
    if [[ ! -e "$directory" ]]; then
        printf 'Restoring missing repository: %s\n' "$directory"
        run /bin/mkdir -p "$(dirname "$directory")"
        run /usr/bin/git clone "$remote" "$directory"
    elif [[ ! -x "$directory/Scripts/build-app.sh" || ! -x "$directory/Scripts/install-app.sh" ]]; then
        printf 'Cannot build %s: expected Scripts/build-app.sh and Scripts/install-app.sh. Existing files were left unchanged.\n' "$directory" >&2
        exit 1
    fi
}

ensure_repository "$FACTORY_REPO" https://github.com/gnaservicesinc/LoomFactory.git
ensure_repository "$WORKSHOP_REPO" https://github.com/gnaservicesinc/LoomWorkshop.git

# Finish every build before touching installed applications. Deliberately avoid
# packaging ZIPs, test builds and any inherited CONFIGURATION=Debug setting.
printf 'Building all three applications for production.\n'
run "${MAKE:-/usr/bin/make}" -C "$ROOT_DIR" CONFIGURATION=Release "DERIVED_DATA=$DERIVED_DATA" build
run "$FACTORY_REPO/Scripts/build-app.sh"
run "$WORKSHOP_REPO/Scripts/build-app.sh"
STUDIO_APP="$DERIVED_DATA/Build/Products/Release/Texture Studio.app"
FACTORY_APP="$FACTORY_REPO/.build/application/Loom Factory.app"
WORKSHOP_APP="$WORKSHOP_REPO/.build/application/Loom Workshop.app"
for application in "$STUDIO_APP" "$FACTORY_APP" "$WORKSHOP_APP"; do
    run /usr/bin/codesign --verify --deep --strict "$application"
done

if [[ "$DESTDIR" == / ]]; then
    # Cocoa asks each app to quit. Busy training apps may refuse; never signal
    # or force-kill them, and do not begin replacing bundles until all are closed.
    run /usr/bin/xcrun swift "$ROOT_DIR/script/suite_app_control.swift" close
fi
printf 'Installing all three applications.\n'
run /usr/bin/xcrun swift "$ROOT_DIR/script/install_macos.swift" "$STUDIO_APP" --destdir "$DESTDIR"
run "$FACTORY_REPO/Scripts/install-app.sh" --skip-build --destdir "$DESTDIR"
run "$WORKSHOP_REPO/Scripts/install-app.sh" --skip-build --destdir "$DESTDIR"

if [[ "$LAUNCH" == true ]]; then
    run /usr/bin/xcrun swift "$ROOT_DIR/script/suite_app_control.swift" launch
    printf 'Texture Studio, Loom Factory and Loom Workshop are installed and running.\n'
else
    printf 'Installed Texture Studio, Loom Factory and Loom Workshop in %sApplications.\n' "${DESTDIR%/}/"
fi
