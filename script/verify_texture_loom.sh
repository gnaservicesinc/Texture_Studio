#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then echo 'usage: verify_texture_loom.sh /path/to/application.app' >&2; exit 2; fi
APP_BUNDLE="$1"
CONTENTS="$APP_BUNDLE/Contents"
if [[ -d "$CONTENTS/Applications" ]]; then echo 'Texture Studio must not contain the retired model training apps.' >&2; exit 1; fi
# Registered Core ML runtime and LoRA source weights are built together as one Texture Loom payload.
UNEXPECTED="$(/usr/bin/find "$CONTENTS" -path "$CONTENTS/Resources/TextureLoom" -prune -o \
  \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' \
     -o -name '*.npy' -o -name '*.npz' -o -name '*.exr' -o -name '*.blend' -o -name '*.mlmodel' \
     -o -name '*.mlmodelc' -o -name '*.mlpackage' -o -name pyvenv.cfg \) -print -quit)"
if [[ -n "$UNEXPECTED" ]]; then echo "Unregistered model or experiment payload: $UNEXPECTED" >&2; exit 1; fi
if [[ -d "$CONTENTS/Resources/TextureLoom" && ! -f "$CONTENTS/Resources/TextureLoom/manifest.json" ]]; then
  echo 'Texture Loom payload is missing its manifest.' >&2; exit 1
fi
