#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then echo "usage: $0 /path/to/application.app" >&2; exit 2; fi
APP_BUNDLE="$1"
CONTENTS="$APP_BUNDLE/Contents"
# This small, user-trained image classifier is the only bundled model. Material
# checkpoints, downloadable depth models and experiment maps stay in user storage.
UNEXPECTED="$(/usr/bin/find "$CONTENTS" \
  \( -path "$CONTENTS/Resources/DiffuseTextureTypeClassifier.mlmodelc" \
     -o -path "$CONTENTS/Applications/Material Review.app/Contents/Resources/DiffuseTextureTypeClassifier.mlmodelc" \
     -o -path "$CONTENTS/Applications/Checkpoint Compare.app/Contents/Resources/DiffuseTextureTypeClassifier.mlmodelc" \
     -o -path "$CONTENTS/Applications/Material Dataset.app/Contents/Resources/DiffuseTextureTypeClassifier.mlmodelc" \
     -o -path "$CONTENTS/Applications/Material Trainer.app/Contents/Resources/DiffuseTextureTypeClassifier.mlmodelc" \) -prune \
  -o \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \
     -o -name '*.onnx' -o -name '*.npy' -o -name '*.npz' -o -name '*.exr' \
     -o -name '*.blend' -o -name '*.mlmodel' -o -name '*.mlmodelc' -o -name '*.mlpackage' \
     -o -name pyvenv.cfg \) -print -quit)"
if [[ -n "$UNEXPECTED" ]]; then
  echo "Application packaging excludes unregistered model weights, numeric experiment maps and runtimes: $UNEXPECTED" >&2
  exit 1
fi
