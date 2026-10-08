# Material learning validation — October 7, 2026

The dataset preparation workflow is implemented and verified. The compact height model trains successfully at native 1K, but its predictions remain too smooth for production material relief.

## Prepared dataset

All 56 materials contribute training images: 112 native 1024×1024 training corners and 56 separate validation regions. Every paired map uses the same integer crop coordinates; no source resizing, numeric gamma correction, per-crop contrast stretching, or storage precision conversion occurs. Validation prefers a disjoint center, then bottom-left or top-right. Fresh unique materials can be collected later to test generalization.

- All 168 generated samples passed independent decoded-pixel comparison with their parent slices.
- The migration audit checked 484 cross-split map rectangle pairs and found no shared pixels.
- Full SHA256 checks confirmed that all 476 existing cropped PNGs remained unchanged.
- Exact previous index/sample metadata and checksums were preserved before changing splits.
- The prior audit checked 242 Poly Haven PNGs including duplicates, with exact published MD5 and byte counts. The selected recipe needs 238 unique Poly Haven parent files. Wicker013's four parent PNGs subsequently matched AmbientCG's official 4K archive exactly by SHA256 and byte count. Wicker013 is Surface Photogrammetry under CC0; it has a native 16-bit displacement map.
- An exact duplicate Plastered Wall 05 folder was archived with its bytes and restoration log intact. Bi Stretch was moved to macOS Trash separately at the user's request.

The current sample directory, including retained manual examples, occupies approximately 2.3 GB. The 4K parents remain available. Deleting them would remove unselected regions and the ability to prepare larger crops.

The wooden-floor center has meaningful partial alpha in 0.289% of diffuse pixels. The explicit `--mask-transparent-input` option preserves straight RGB and original height, excluding those pixels and an eight-pixel margin from loss/metrics. It retains 98.13% eligible pixels in that crop. This is a loss-validity mask, not inferred confidence or proof of completely isolated network context. All 112 training crops pass without meaningful transparency.

## Actual learning run

`native-height-region-pilot-02` trained a 218,209-parameter U-Net from scratch on Metal for 1,200 native-1K optimizer steps, taking 404.28 seconds after initial evaluation. This is a compact material-height pilot, not a DA3 GIANT LoRA. Source PNGs remain at their original integer precision; only active targets become Float32 values divided by 65535. The loss uses offset-invariant height, multiscale finite differences, and native radius-4 high-pass detail. Offset removal occurs only inside the loss.

| Metric | Training crops | Separate regions |
| --- | ---: | ---: |
| Samples evaluated | 112 | 56 |
| Native gradient error improvement over flat height | 1.38% | 1.25% |
| High-pass error improvement over flat height | 1.88% | 1.62% |
| Mean per-sample prediction/target gradient-energy ratio | 0.156 | 0.143 |

These energy ratios average each sample's ratio; they are not globally pooled energy ratios. Scalar absolute-height origin is unconstrained by the offset-invariant objective. The model learns weak cloth weave and some seams, but soil and stucco remain largely flat even on training inputs. Thus the immediate issue is fitting the expected detail. This experiment does not establish new-material or casual-photo performance.

Numeric predictions are lossless ZIP-compressed FLOAT32 EXRs and matching Float32 NPY arrays, with exact EXR decoding checks. Derived normals use OpenGL +Y. Separate display previews use fixed gains; they never replace numeric data. Blender reads diffuse as sRGB and numeric maps as Non-Color. Earlier real Blender readback of height and normal EXRs matched their Float32 arrays exactly.

## Native 2K capacity

The same compact model's actual Float32 forward/backward/optimizer probe passed at batch one:

| Native crop | Mean compute step | Sampled Metal driver allocation |
| --- | ---: | ---: |
| 1024×1024 | 0.160 s | 2.19 GB |
| 2048×2048 | 0.598 s | 8.18 GB |

2K is practical on this 64 GB Mac for this architecture, approximately 3.75× slower in measured compute. Decoding, validation, and export add time. This does not establish GIANT LoRA training capacity or improved 2K output. Metal driver allocation and process RSS can overlap and must not be summed.

## Artifacts and rerunning

Run `bash /opt/ipde/ipde/scripts/prepare-materials.sh /opt/ipde/material-dataset` after adding sources. An optional `2048` argument prepares a separate `prepared-2048/` dataset. When two training corners leave no disjoint validation space, use one training corner and the opposite validation corner; sources too small for two disjoint crops are reported.

- Workflow: [material-training-workflow.md](material-training-workflow.md).
- Dataset: `/opt/ipde/material-dataset/dataset.json`.
- Final audit: `/opt/ipde/ipde/out/material-training/region-split-final-audit.json`.
- Source-pixel verification: `/opt/ipde/ipde/out/material-training/region-crop-verification.json`.
- Full run: `/opt/ipde/ipde/out/material-training/native-height-region-pilot-02/`.
- Durable checkpoint, frozen metadata, reports, and comparison sheets: `/opt/ipde/material-dataset/experiments/native-height-region-pilot-02/`.

The checkpoint remains an experiment. No automatic model promotion occurred. The initial dataset refresh passed 73 relevant Python regression tests, compilation checks, shell syntax validation, and Git whitespace checks. Subsequent fitting, pretrained comparison and source-recreation work is recorded in [the fitting report](material-fitting-2026-10-07.md).
