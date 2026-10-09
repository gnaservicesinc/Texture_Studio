# Texture Studio

A standalone SwiftUI app for Apple Silicon that turns a surface photograph into an editable Blender Cycles material. The app uses ImageIO, Core Image, Metal and a local Python/PyTorch material-model backend. It requires macOS 26 and full Xcode 26 to build.

## Build and launch

```sh
make build
./script/build_and_run.sh
```

Open `native/TextureStudio/TextureStudio.xcodeproj` and select the shared `TextureStudio` scheme. Release is the default configuration. `make test-native` runs native regressions; `make smoke` exercises EXR formats. `make package` produces an ad-hoc signed local archive. Packaging does not publish or notarize the app.

**Model Training** opens the dataset, trainer, checkpoint and review workspaces. Material Review, Checkpoint Compare, Material Dataset and Material Trainer also launch independently from the app bundle. See [native material tools](docs/native-material-tools.md).

## Photo to material

1. Import a HEIC, JPEG, PNG, TIFF or Apple-supported RAW surface photograph.
2. Correct perspective and framing. Camera metadata can guide focal settings; unknown calibration remains an estimate.
3. Prepare a balanced diffuse map using broad illumination correction and color processing. Preserve fine detail. Registered companion photographs may reduce capture noise; the photo path applies no spatial denoiser.
4. Generate material maps with a selected refined material checkpoint, attach a registered surface-height map, or keep flat displacement. Embedded portrait depth does not supply material relief.
5. Review diffuse, roughness, normal and displacement. Set material width and displacement scale and save the recipe.
6. Export maps and the Blender setup script into a new folder. Generation/export size remains independent of the training size.

| Map | Export | Blender interpretation |
| --- | --- | --- |
| Diffuse | 16-bit sRGB PNG | sRGB base color |
| Roughness | 16- or 32-bit float EXR | Non-Color scalar |
| Normal | 16- or 32-bit float EXR | Non-Color, tangent-space OpenGL +Y |
| Displacement | 16- or 32-bit float EXR | Non-Color relative height |

Numeric map exports receive no photographic gamma or tone mapping. Float32 export preserves the current Float32 samples; it cannot recover detail absent from a source or prediction. Original files and precision-preserving HEIF auxiliary exports remain unchanged.

## Refine a material model

Training uses complete registered diffuse and target maps at one explicit grid. Every input, target and training review matches that grid; no smaller random crop is hidden inside the loader. Supported training sizes are filtered by source pixels and the machine's memory budget. A test photo passes through the same diffuse preparation as application generation before reaching the model.

Exact-size originals are referenced directly. Only intentional rescaling creates final temporary training images. Changing size, finishing training or stopping a run removes those owned files while preserving original sources and small per-size review records. The local `/opt/ipde/material-dataset` collection now references 97 original material sets without duplicating their image files.

The experimental backend uses a complete pinned PBRnxt material network adapted to native output scale. It refines height, roughness or normal with LoRA. Enable **Developer mode** in Settings to expose model controls and export a full fused `.safetensors` checkpoint plus the separate LoRA. Normal mode exports the LoRA. Compatible adapters can be combined with explicit weights against the same base and target modules.

Saved models can be uploaded to Hugging Face using the selected account, repository and visibility. Developer mode offers upload after training. Successfully uploaded models appear in the saved Hub catalog with a download action. App-owned base models can be removed and downloaded again from their recorded origin.

The optional local Clef decision model can advise on photo preparation, dataset suitability and output detail, appeal and artifacts. Review recommendations remain editable. Full-resolution maps and displaced surfaces determine acceptance; a lower loss or an adviser score does not automatically select a shipped model.

See [training workflow](docs/material-training-workflow.md), [data contract](docs/material-training-data.md), [quality acceptance](docs/material-model-vetting.md), and [model setup](docs/model-setup.md).

## Precision-preserving extraction

The retained Python extraction tools use `pillow-heif>=1.5.0`, inspect actual source precision and preserve raw depth and auxiliary image data independently of the artistic material workflow.

```sh
make setup
PYTHONPATH=src .venv/bin/python ipde_extract.py --help
make test-python
```

Datasets, model weights, generated outputs and app bundles stay outside Git. This repository is in development before V1; removed experiments are not released features and have no compatibility support in the material app. See [release requirements](docs/releasing.md) and [license](LICENSE).
