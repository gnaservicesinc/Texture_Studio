#!/usr/bin/env bash
# Explicit, revision-pinned downloads. Inference itself is always local.
set -euo pipefail

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_REPO_DIR="$(cd "$TASK_SCRIPT_DIR/.." && pwd)"
TASK_PREFIX="$(cd "$TASK_REPO_DIR/.." && pwd)"
TASK_PYTHON="$TASK_REPO_DIR/.venv/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { echo "--prefix requires a directory" >&2; exit 2; }
            TASK_PREFIX="$2"; shift 2 ;;
        --python)
            [[ $# -ge 2 ]] || { echo "--python requires a Python executable" >&2; exit 2; }
            TASK_PYTHON="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--prefix DIR] [--python PYTHON] [depthpro|depth-anything-v2|depth-anything-v2-small|depth-anything-3 ...]"
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [[ $# -eq 0 ]]; then
    set -- depthpro depth-anything-v2
fi
for TASK_MODEL in "$@"; do
    case "$TASK_MODEL" in
        depthpro|depth-anything-v2|depth-anything-v2-small|depth-anything-3) ;;
        *) echo "Unknown model: $TASK_MODEL" >&2; exit 2 ;;
    esac
done
"$TASK_PYTHON" -m pip install -r "$TASK_REPO_DIR/requirements-depth.txt"
TASK_HF="$("$TASK_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts") + "/hf")')"
mkdir -p "$TASK_PREFIX/models"

checkout_source() {
    local task_url="$1" task_dest="$2" task_revision="$3"
    if [[ -d "$task_dest/.git" ]]; then
        if [[ -n "$(git -C "$task_dest" status --porcelain)" ]]; then
            echo "Preserving edited model source: $task_dest. Commit/stash it or choose another --prefix." >&2
            exit 1
        fi
        if [[ "$(git -C "$task_dest" rev-parse HEAD)" == "$task_revision" ]]; then
            return
        fi
    elif [[ -e "$task_dest" ]]; then
        echo "Preserving existing non-Git directory: $task_dest. Choose another --prefix." >&2
        exit 1
    else
        git clone --no-checkout --filter=blob:none "$task_url" "$task_dest"
    fi
    git -C "$task_dest" fetch --depth 1 origin "$task_revision"
    git -C "$task_dest" checkout --detach "$task_revision"
}

for TASK_MODEL in "$@"; do
    case "$TASK_MODEL" in
        depthpro)
            checkout_source https://github.com/apple/ml-depth-pro.git "$TASK_PREFIX/ml-depth-pro" 9e65e4dbe9568d23c546fcec53302b10445e109e
            HF_HUB_DISABLE_XET=1 "$TASK_HF" download apple/DepthPro depth_pro.pt --revision ccd1350a774eb2248bcdfb3be430e38f1d3087ef --local-dir "$TASK_PREFIX/models"
            ;;
        depth-anything-v2|depth-anything-v2-small)
            checkout_source https://github.com/DepthAnything/Depth-Anything-V2.git "$TASK_PREFIX/Depth-Anything-V2" a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
            if [[ "$TASK_MODEL" == depth-anything-v2 ]]; then
                HF_HUB_DISABLE_XET=1 "$TASK_HF" download depth-anything/Depth-Anything-V2-Large depth_anything_v2_vitl.pth --revision cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9 --local-dir "$TASK_PREFIX/models"
            else
                HF_HUB_DISABLE_XET=1 "$TASK_HF" download depth-anything/Depth-Anything-V2-Small depth_anything_v2_vits.pth --revision 03876f8651c73a60fe4c2c48294e09fcb6838fcf --local-dir "$TASK_PREFIX/models"
            fi
            ;;
        depth-anything-3)
            checkout_source https://github.com/ByteDance-Seed/Depth-Anything-3.git "$TASK_PREFIX/Depth-Anything-3" 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
            HF_HUB_DISABLE_XET=1 "$TASK_HF" download depth-anything/DA3-GIANT-1.1 config.json model.safetensors --revision 72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19 --local-dir "$TASK_PREFIX/models/DA3-GIANT-1.1"
            ;;
    esac
done
echo "Local models prepared under $TASK_PREFIX. Use explicit model_path/source_dir when this prefix differs from the repository parent."
