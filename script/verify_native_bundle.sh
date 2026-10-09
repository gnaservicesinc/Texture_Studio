#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then echo "usage: $0 /path/to/application.app" >&2; exit 2; fi
APP_BUNDLE="$1"
if [[ ! -d "$APP_BUNDLE/Contents" || -L "$APP_BUNDLE" ]]; then
  echo "A complete native application bundle is required: $APP_BUNDLE" >&2; exit 1
fi
UNEXPECTED="$(/usr/bin/find "$APP_BUNDLE/Contents" \( -iname '*.py' -o -iname '*.pyc' -o -iname '*.pyo' \
  -o -iname 'libpython*' -o -iname 'python*.framework' -o -name pyvenv.cfg \
  -o -name .venv -o -name venv -o -name __pycache__ -o -name '*.egg-info' \
  -o -name MaterialBackend \) -print -quit)"
if [[ -n "$UNEXPECTED" ]]; then
  echo "Native application contains a retired interpreter or backend resource: $UNEXPECTED" >&2; exit 1
fi
