# Curated material training data

This is the data contract and original source-header audit. The [preparation and native height-training scripts](material-training-workflow.md) now implement crops, verification, metadata and a local supervised pilot. The dataset lives at `/opt/ipde/material-dataset`, with original `sources/<asset>/` and generated matched maps under `samples/<asset>_auto_<crop>/`. Keep the source's actual format and precision; no global conversion is required. Original high-resolution photographic Poly Haven maps are the default source. Human review selects assets and crops; screenshots are visual references, never numerical training targets.

## Source precision

Prefer original **16-bit PNG displacement and OpenGL normal** downloads when available. PNG uses lossless compression, but its extension alone does not establish sample depth: inspect the header and decoded array. Preserve height as uint16 and normals as three uint16 channels in storage. Avoid JPEG, web previews, 8-bit conversion, gamma correction and automatic contrast stretching for these numeric maps. Color inputs have their own declared color-space handling. [PNG specification](https://www.w3.org/TR/png-3/).

The current `white_stucco_02` headers were checked at 1K/2K/4K/8K. Displacement PNG is 16-bit grayscale; normal PNG is 16-bit RGB. EXR alternatives contain HALF channels but use DWAA compression at level 45. OpenEXR's default rules subject these Y/R/G/B channels to lossy DCT. A high bit-depth EXR is therefore not automatically a lossless target. Header inspection does not measure either file's accuracy against the author's original capture. Preserve the downloaded original and record the representation used. [Asset](https://polyhaven.com/a/white_stucco_02), [header evidence](data/white-stucco-02-header-audit-2026-10-07.json), [OpenEXR DWA channel rules](https://raw.githubusercontent.com/AcademySoftwareFoundation/openexr/main/src/lib/OpenEXRCore/internal_dwa_classifier.h).

For bounded 0–1 maps, uint16 has 65,536 evenly spaced codes. HALF uses a 10-bit fraction and floating exponent; it can have substantially coarser spacing around midrange values. Converting an original HALF sample to Float32 preserves it exactly. Converting uint16 to Float32 also preserves every integer code; a separately declared divide by 65535 supplies the model's 0–1 target. This changes representation without creating detail. Keep the original codes for verification. [OpenEXR sample types](https://openexr.com/en/latest/TechnicalIntroduction.html#the-half-data-type).

The decoder must preserve 16-bit RGB normals as well as scalar height. Acceptance requires checking the actual decoded dtype/channel order, reading deliberately adjacent source codes, and round-tripping without merged levels. A decoder that silently returns 8-bit RGB is rejected. A crop must compare exactly with its parent's corresponding integer array slice.

## Dataset storage and model arithmetic

Keep compressed 16-bit originals on disk. Store crop rectangles and review records as a small manifest; decode/crop the active sample on demand, with a bounded decoded-parent cache. Materialize cropped PNGs only if measured loading performance justifies the extra storage. Do not create a permanent Float32 copy of every map merely because the model uses Float32 tensors.

Storage precision, target precision, model arithmetic and optimizer precision are distinct. Convert the current batch to Float32 in memory; use Float32 height/gradient/normal loss calculations to avoid discarding small differences. Mixed-precision model inference/backpropagation can be evaluated independently where the selected Mac backend supports the necessary operations. A 16-bit source is not a requirement to calculate every operation in FP16. [PyTorch mixed precision](https://docs.pytorch.org/docs/2.14/amp.html).

Uncompressed single-channel height sizes:

| Crop | uint16 source samples | Float32 tensor |
| --- | --- | --- |
| 1024×1024 | 2 MiB | 4 MiB |
| 2048×2048 | 8 MiB | 16 MiB |

RGB normals take three times these amounts. PNG disk compression varies with the material. Activations, gradients, optimizer state and the model usually dominate training memory; these target-array sizes cannot establish whether a model will fit in 64 GB.

## Native crops and manual review

1. Check every paired map's true dimensions and UV registration. Select 4K/8K sources when their native detail is useful; do not resize the full map before extracting the requested crop.
2. A genuine 2048×1024 source yields two non-overlapping 1024×1024 crops. A square 8192×8192 source has up to 64 non-overlapping 1K crops or 16 2K crops, but only manually accepted crops enter training. Resolution labels are not dimension checks. The current White Stucco 02 files are square; the generic rectangular-material rule still applies.
3. Use the same integer crop rectangle for color, height, normal and roughness. Show a linked crop at native zoom, its height/normal previews and a neutral/grazing render. Record accept/exclude, reason, category and quality concerns. Retain rejected candidates for reversible review.
4. Store asset ID, original URLs, locally verified full-file hashes, channel types, encoding, pixel dimensions, compression, crop rectangle, source resolution, normal convention and verified physical scale. Keep unreliable scale metadata marked unknown. White Stucco 02 currently has an implausible API width; it must not set physical crop size automatically.
5. A crop from an 8K source covers a smaller part of the physical surface. Keep its footprint and texel density when known; use multiple scales for fine grain and larger structure. Preserve the parent height offset/scale rather than stretching every patch independently.
6. The current learning pilot uses every source material for training and reserves a different, non-overlapping native region from each source for validation. This measures progress on familiar materials; fresh material identities will be collected separately when testing generalization. The optional whole-material split remains available for that later assessment. Rotations/flips must transform tangent normal components as well as pixel positions; plain image flips can create incorrect normal targets.

Start with a small, manually approved set. The planned trainer should report an actual batch-one forward/backward memory probe at each proposed crop size before a longer run. Native 1K is the first meaningful detail target; 2K is conditional on measured model memory. LoRA or a residual decoder changes the learning method, not the storage contract or the quality of the source maps.

The original format audit fetched only bounded 32 KiB prefixes; those prefix hashes are not whole-file checksums. Later preparation uses the user's complete downloaded maps and verifies actual full-file hashes and decoded crop pixels. See the workflow and dated run reports for the subsequent dataset and training evidence.
