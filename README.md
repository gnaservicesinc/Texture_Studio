# Texture Studio

A standalone SwiftUI app for Apple Silicon, turning a real surface photograph into an editable Blender Cycles material. Requires macOS 26 or later and full Xcode 26 or later to build. The active app uses SwiftUI, Liquid Glass, ImageIO, AVFoundation, Vision, Core Image, Metal and Core ML. Qt and the CMake application build have been removed.

## Build and launch

```sh
make build
./script/build_and_run.sh
```

Open `native/TextureStudio/TextureStudio.xcodeproj` in Xcode and choose the shared `TextureStudio` scheme, or use the Codex Run action. The default local app is `build/TextureStudio/Build/Products/Release/Texture Studio.app`. `make test-native` runs native regressions; `make smoke` exercises both EXR formats. `make package` produces an ad-hoc signed archive; this is not notarization or publication. There are no Windows or Linux application build targets.

Use **Model Training** in Texture Studio’s toolbar or sidebar for a guided workspace: prepare native crops, start from a base or saved model, compare checkpoints, inspect full-resolution details, and export or use a model in Studio. **Material Review**, **Checkpoint Compare**, **Material Dataset**, and **Material Trainer** are also bundled as independently launchable apps inside Texture Studio. Normal builds and Run actions use Release. Changing the training crop size automatically prepares matching crops from original maps. See [native material tools](docs/native-material-tools.md) for launch commands and checkpoint selection.

## Photo to material

1. Import a HEIC, JPEG, PNG, TIFF or Apple-supported RAW photograph of a surface. Camera, lens and auxiliary metadata are displayed. Spatial companion photographs come from ImageIO stereo groups, rather than arbitrary auxiliary images.
2. Adjust X/Y tilt and Z rotation around the image center. The square crop is computed inside the projected footprint. Crop scale and position select a tighter area. Override focal length when needed; missing calibration is identified as an estimate. Manual radial correction handles modest lens distortion.
3. Balance broad illumination and reduce noise conservatively. Preview the changes rather than assuming a perfect separation of lighting and reflectance.
4. Generate depth with the exact DA3-GIANT-1.1 PyTorch/MPS model, or attach a registered map. Embedded portrait depth never supplies material relief. Spatial companions are aligned and exposure/color matched; only regions passing agreement checks contribute. Occlusions, parallax and weak matches keep the primary photograph.
5. Review diffuse, roughness, normal and displacement separately. Set material width and displacement scale. Save a JSON recipe to repeat the edits.
6. Export a new material folder. Default size is 1024 square; 2048, the requested 4098, and 8192 are available. Previews and inference stay small independently of export size. Large exports require acknowledgement and pass a memory budget check.

| Map | Export | Blender interpretation |
| --- | --- | --- |
| Diffuse/base color | 8-bit sRGB PNG | sRGB, Base Color |
| Roughness | 16- or 32-bit float EXR | Non-Color, scalar |
| Normal | 16- or 32-bit float EXR | Non-Color, tangent-space OpenGL +Y, RGB encoded 0–1 |
| Displacement | 16- or 32-bit float EXR | Non-Color, relative height 0–1, midlevel 0.5 |

Exports include `material.json`, setup notes, and a Blender Python script that creates shader nodes without changing geometry. All maps share the crop and UV layout. Originals and extracted scientific data are preserved. Float32 storage does not recover lost camera or FP16 model precision.

## Local models

Missing models offer **Locate**, **Download**, or **Continue with Flat Surface**. Managed downloads live in `~/Library/Application Support/Texture Studio/Models`; **Remove** deletes those downloads. Located external files are unlinked without deletion. Downloads are revision pinned, SHA-256 checked and staged, with progress, cancellation and retry. Model validation follows the selected backend.

The default is [Depth Anything 3 GIANT 1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1), running locally through PyTorch/MPS. It is not converted to a smaller Core ML model. Choose an explicit 1036, 1540 or 2044 pixel inference edge independently of export size; the app does not attempt 5K transformer inference or silently change the selected size. Its weights are **CC BY-NC 4.0, non-commercial**. The model estimates relative scene depth, which is converted into relative material height with robust plane removal and depth-only artifact cleanup. Near-flat protection prevents amplifying tiny predicted variations into strong bumps and can be disabled. Raw model samples remain untouched; brightness-derived relief is off by default. Compatible custom Core ML models remain an optional backend, with explicit output selection when ambiguous. The retired Apple Small V2 model is removed from the active catalog.

**Local Models** also installs a managed PyTorch/DA3 environment or locates an existing Python executable. The native app includes its own worker and does not communicate with the old studio. Runtime removal deletes only the managed environment; external environments are unlinked. Inference cancellation terminates the worker and releases its model memory.

**Photo Review** uses exactly [clef:27b-nvfp4](https://ollama.com/library/clef:27b-nvfp4), the optional 18 GB MLX decision model, through local Ollama. [Ollama v0.40.0](https://github.com/ollama/ollama/releases/tag/v0.40.0) adds MLX decision support and automatically uses MLX for supported architectures on Apple Silicon. Texture Studio requires that version or newer, validates the exact tag's format/quantization and capabilities, and never silently substitutes GGUF. The adviser chooses among predefined lighting, noise, relief and roughness options; suggestions require review and **Apply**. Photos go to numeric loopback only as bounded 768-pixel copies. `keep_alive: 0` releases model memory after requests. Pull/cancel/delete controls identify the large download and shared Ollama model explicitly.

## Scope and quality

This native workflow produces artistic photo-based material estimates. Depth supplies relief; embedded effects depth and printed colors do not become geometry by default. Material Trainer fits native height, roughness and OpenGL-normal heads with a pinned DINOv2 encoder. Existing frozen and LoRA checkpoints can be compared; the interactive trainer currently fits frozen-encoder heads. Texture Studio can use an explicitly selected height checkpoint, preserving its learned amplitude. Roughness/direct-normal checkpoint integration in Studio, automatic hard-shadow reconstruction and guaranteed seamless tiling remain future work. Review in neutral Cycles lighting before production use. See [workflow](docs/texture-studio.md), [native material tools](docs/native-material-tools.md), and [material model/refinement research](docs/material-refinement-research.md).

RAFT research remains a separate Python tool; it is not part of the new photo-material workflow. Historical Qt documentation lives under `docs/legacy`. Existing Python extraction/dataset/training code and local data/model folders are preserved for optional research interoperability. The material tools bundle their Python source backend; models, datasets and Python environments are managed separately.

For the retained precision-preserving extraction CLI:

```sh
make setup
PYTHONPATH=src .venv/bin/python ipde_extract.py --help
make test-python
```

See [model setup](docs/model-setup.md), [surface validation](docs/texture-surface-validation-2026-10-07.md), [release requirements](docs/releasing.md), and [license](LICENSE).
