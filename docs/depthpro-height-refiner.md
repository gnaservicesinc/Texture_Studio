# DepthPro Height Refiner

**DepthPro Height Refiner** is a separate native restoration experiment. It takes one prepared DepthPro depth channel and predicts one material-height channel. The original diffuse photograph generates the DepthPro map beforehand; the refiner receives no RGB channels. This differs from **Compact scalar**, which learns the complete RGB-to-height mapping.

Inference runs DepthPro once for the prepared source field, then the refiner. Training prepares and caches DepthPro predictions, then unloads the large model before optimizing the small network, so it does not run inside every training epoch. The refiner trains all its weights with Apple MPSGraph Float32 and uses the same directly editable optimizer controls as the other compact families.

## Pairing and numeric preparation

Apple's [reference implementation](https://github.com/apple-aiml-research/ml-depth-pro) predicts camera depth in meters. Camera distance and relative material height have different direction, scale and meaning. The experiment uses an explicit shared preparation contract rather than treating a display grayscale image as a numeric target.

1. Prepare the diffuse field and original height target with the dataset's recorded exact grid and crop coordinates. Run DepthPro on that diffuse field, preserving the entire field of view.
2. Restore the raw meter prediction to the prepared field's exact grid using pixel-center bilinear interpolation in top-left row-major coordinates. Cache the original floating-point meter tensor separately from this preparation.
3. Compute reciprocal depth, `inverse = 1 / meters`, then normalize over this complete field: `input = (inverse - minimumInverse) / (maximumInverse - minimumInverse)`. Larger input values indicate nearer predicted surfaces. Compute the extrema independently for each field before training, and use the same rule at inference. A constant field produces all-zero input.
4. Decode the paired original integer height map directly as `target = originalCode / codeMaximum`. Preserve its direction, integer values and crop alignment; do not normalize its range to match DepthPro.

Full-field normalization removes absolute camera scale. It does not supply detail absent from the DepthPro prediction or establish an absolute physical displacement scale. The network learns the dataset's original relative-height target convention.

The cache stores both the original raw Float32 meter tensor and the normalized Float32 input at the paired grid in safetensors, together with source identity, dimensions, exact DepthPro model identity and preparation contract `texture-studio-depthpro-inverse-depth-v1`. A changed source, model or contract cannot reuse a cache entry as if it described the same training pair. Display previews remain separate from these numeric arrays and from the original height maps.

The training [crop timing and orientation controls](material-training-workflow.md) also apply to this family. Identical prepared inputs reuse their verified cache. Orientation-only flips permute the cached depth field to match the crop, so they do not rerun DepthPro. A newly spun center field requires a fresh prediction before any optional flip. Spin starts from the original source, while progressive flips transform the current cached crop, and both keep the height target spatially aligned.

## Native DepthPro adapter

The [Core ML conversion](https://huggingface.co/coreml-projects/DepthPro-coreml) requires an RGB `image` at 1536×1536 and `originalWidth` with shape `[1,1,1,1]`, and returns `depthMeters` with shape `[1,1,1536,1536]`. Texture Studio supplies both inputs through a dedicated native adapter. `originalWidth` describes the prepared source field before the adapter's 1536×1536 resize. Core ML embeds the image normalization; the adapter preserves the full field without additional display normalization or tone mapping.

The adapter loads the model once with `MLComputeUnits.all`; Core ML chooses the available execution devices. The pinned conversion is `coreml-projects/DepthPro-coreml` revision `e3b374e875fa7c89ca4c80203e1fb12e5e02bceb`. Its width input and meter output are Float16. Converting the returned numbers to Float32 for caching and refiner work preserves those returned values but does not recover precision lost inside the upstream conversion. The small refiner's tensors, gradients and optimizer state remain Float32.

In Model Training, select **DepthPro Height Refiner** and start the operation. A missing model selection automatically acquires the pinned Core ML package into `~/Library/Application Support/Texture Studio/DepthPro/DepthPro.mlpackage`. Setup shows byte progress and supports cancellation. It verifies all three files against their pinned sizes and SHA-256 values, then checks the complete package identity before publishing it. Verified staging files are reused on retry; concurrent windows share one installation lock. A verified installed package works offline. The download is 1,902,515,337 bytes. Apple's `apple/DepthPro-hf` Transformers weights are a different format; the native adapter obtains the compatible conversion directly without an interpreter or conversion step.

**Locate DepthPro…** and **DepthPro Core ML package** also accept an existing developer-selected model. Native train/infer operations accept `--model-family depthpro-height-refiner`, `--depthpro-model` with that model path, and optional `--depthpro-cache` to choose the numeric cache directory. Missing custom paths fall back to managed storage, without creating files at the missing selection. The default cache is `~/Library/Caches/Texture Studio/DepthPro Inputs`. A refiner checkpoint records the exact package SHA-256 and preparation contract; comparison, warm starts and inference require the matching upstream model. Refiner exports include the complete small network and its configuration, while DepthPro remains a separate local dependency. RGB compact checkpoints, PBRnxt adapters and DepthPro refiner checkpoints have distinct architecture and input contracts and are not interchangeable.

## Evaluation

The small network may learn systematic depth errors, recurring shapes and boundary corrections with less work than learning material shape entirely from RGB. Missing detail remains ambiguous: inferred relief may be plausible without matching the real material. One input/output channel reduces boundary work, but internal feature layers dominate the compact network, so this change alone does not imply a threefold speedup.

Reserve whole materials outside training, retaining all their related crops and diffuse variants in the same holdout group. Compare raw prepared DepthPro input, refined height, the original height reference and rendered displacement at matching coordinates. Judge new-material structure, boundaries and fine relief separately from training loss. A loading smoke test or successful optimizer step establishes pipeline functionality; it does not establish useful learned detail, 1024 training quality or production qualification.

Checkpoint Compare permits one selected checkpoint without the optional untrained initialization. Its output can be inspected alone or beside a recorded dataset height reference. The initialization can also run by itself; only an empty checkpoint selection with the initialization unchecked lacks a model to run.

## Upstream licensing status

Checked on 2026-10-11, the conversion card lists `apple-ascl`, and Apple's [GitHub sample-code license](https://github.com/apple-aiml-research/ml-depth-pro/blob/main/LICENSE) is permissive. Apple's current [official model license on the Hub](https://huggingface.co/apple/DepthPro/blob/main/LICENSE) instead restricts the model and its derivatives to noncommercial research and excludes product development and commercial products or services. These sources disagree about the upstream weights' applicable terms. Treat this integration as a research experiment; commercial model rights and redistribution are unresolved by the conversion card alone. Exporting only the small refiner does not establish permission to use the upstream DepthPro weights commercially.

The [build 109 assessment](validation/compact-material-fixes-and-1k-trial-2026-10-11.md#depth-only-refiner-assessment) records the earlier proposal on 2026-10-11. Its statement that no DepthPro weights were downloaded and no new refiner training was started applies to that historical assessment, before this separate implementation.

The [build 110 validation report](validation/depthpro-and-balanced-crops-2026-10-11.md) records the real adapter and refiner trials, production numeric readback, crop regressions and installed-app checks.
