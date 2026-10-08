# Material quality, review, and training

**Historical DINOv2 experiment workflow.** DINOv2 training and production Studio inference are retired. Current dataset preparation and review are documented in [native material tools](native-material-tools.md); replacement models must pass [model vetting](material-model-vetting.md). The commands and measurements below preserve earlier experiments, not a current production recommendation. Some referenced local runs may have been removed during cleanup.

Archival `material_training_cycle.py train`, `resume` and `probe` commands now require **`--allow-retired-experiment`**. Add it after the subcommand when deliberately reproducing an older experiment. Without it, the CLI rejects the launch before checkpoint loading, dataset setup or model allocation. Opting in prints a retirement warning; it does not make the result eligible for Studio activation.

The goal is useful, realistic materials made from a photograph. Height-reference error, correlations, and gradient energy are diagnostics, not a quality verdict. Select candidates by visible relief, preservation of useful surface detail, noise/artifacts, plausible roughness, and how the maps work together. Do not promote a checkpoint automatically because its loss is lower.

## What to inspect

Put the original diffuse photo on genuinely displaced geometry and tilt the view. Look for stones or ridges sinking into the surface, shadows becoming cavities, implausible protrusions, and inverted relief. Use the same displacement gain, camera, light, and roughness for every candidate. A flat-height control reveals what the photo alone contributes. A clay view exposes relief that the diffuse color can hide. Grazing light makes fine structure and unwanted noise easier to inspect.

The source reference is a useful guide. Pixel equality with it is not the objective. Record which candidate you would use and why; a pleasant smoother surface may beat an overly noisy “more detailed” surface. Inspect at the actual intended material scale and final output resolution. Inspect the borders separately for tiling: these cropped examples are not guaranteed seamless.

Geometry preview sampling and rendered output pixels are separate from training resolution. A reduced mesh preview diagnoses large bumps and pits; it cannot prove that every native texel's fine relief is preserved. Numeric maps remain native precision and resolution.

## Native 2K preparation

Run commands from `/opt/ipde/ipde`. This bounded pilot takes native 2048-square regions from the original 4K parents, preserving the full 97-material 1K index. It is a separate experiment: its review rectangles differ from the older 1K center rectangles, so its metrics cannot be compared as if the test images were identical.

```sh
.venv/bin/python scripts/material_dataset.py prepare-region-splits \
  --sources /opt/ipde/material-dataset/sources \
  --dataset /opt/ipde/material-dataset/pilots/native-2k-four-material-01 \
  --crop-size 2048 \
  --material white_stucco_02 --material farm_soil \
  --material cotton_jersey --material broken_brick_wall
```

Repeat `--material` for other source families. Both training corners are used when a disjoint third region fits. A 4K square supplies two 2K training corners and a bottom-left review region; a 4K rectangle may supply only one training corner and an opposite review region. Existing sample maps are verified and reused. A selection cannot omit materials from an existing index; create a new subset directory instead.

The prepared four-material pilot contains 8 training crops and 4 disjoint review crops, about 752 MiB for all original-precision maps. Its 12 samples were independently decoded and compared against the corresponding source regions. Larger native crops include more original surface pixels; this is different from training on the provider's scaled 2K view of the whole material. Do not mix provider resolutions into the historical split without checking their shared surface footprints.

## Reproduce archived training

```sh
.venv/bin/python scripts/material_training_cycle.py train --allow-retired-experiment \
  --dataset /opt/ipde/material-dataset/pilots/native-2k-four-material-01 \
  --target height --expected-size 2048 --updates-per-crop 100 \
  --warm-start out/material-training/four-material-adaptation-01/frozen/checkpoint.final.pt \
  --selection final --prediction-limit 4 --max-minutes 15 \
  --output out/material-training/my-2k-height-01 --allow-unreviewed
```

This balances every selected training corner, caches only the coarse frozen features, and trains one native RGB/target pair at a time. Only the encoder's contextual view is resized to 518; the supervised head and numeric targets stay 2048. `--material` can select a subset of a larger prepared index. A fresh warm start explicitly resets AdamW. `--allow-unreviewed` means the generated sources have not received a manual sample approval; their numerical verification still runs.

Checkpoints retain optimizer state, exact crop/file identities, target type, encoder pins, and the balanced schedule. Ctrl+C saves completed updates. Resume into a **new** output directory:

```sh
.venv/bin/python scripts/material_training_cycle.py resume --allow-retired-experiment \
  --resume-checkpoint out/material-training/my-2k-height-01/checkpoint.latest.pt \
  --output out/material-training/my-2k-height-01-resumed
```

The command inherits the saved training settings and refuses changed maps or incompatible targets. To continue a completed fit, explicitly increase `--updates-per-crop`; the original schedule must remain an exact prefix. Do not switch resolution during an optimizer resume. Use a fresh `train --warm-start` for an intentional resolution change.

For separate roughness and direct normal heads, run the same training command with `--target roughness` or `--target normal` and a new output directory. Omit `--warm-start` for a fresh head. The frozen encoder remains shared on disk; the learned output and supervision are separate. Normal targets use the actual OpenGL vector labels, not a gradient of height. These commands make those experiments possible; they do not claim production quality before a visible review.

The native-size probe also runs actual forward/backward/AdamW steps:

```sh
.venv/bin/python scripts/material_training_cycle.py probe --allow-retired-experiment \
  --dataset-1024 /opt/ipde/material-dataset \
  --dataset-2048 /opt/ipde/material-dataset/pilots/native-2k-four-material-01 \
  --warm-start out/material-training/four-material-adaptation-01/frozen/checkpoint.final.pt \
  --steps 3 --output out/material-training/my-size-probe --allow-unreviewed
```

On this M2 Max/64 GB Mac, the current head measured 0.163 seconds/update at 1K and 0.611 at 2K, with sampled Metal driver allocations of 2.60 and 8.56 GB. This is a short feasibility test, not an OS-wide memory-pressure measurement or output-quality result. That historical run used sampled time and 30 GB driver guards; an in-flight allocation can occur before the next guard check.

## Historical 2K round and reproduction steps

The first native 2K round completed **800 updates**, 100 on each of the eight training corners, in **827.68 seconds** including input verification, evaluation and export. Sampled Metal driver allocation peaked at **8.79 GB**. The full training cycle is slower than the short compute probe because it loads and verifies native maps for each update. Four disjoint 2K regions were exported without resizing. This round used a frozen DINOv2 encoder and trained the material head; it did not retrain LoRA. Larger crop coverage, both corners, and further updates changed together, so this experiment does not isolate resolution as the sole cause of improvement.

The review is `/opt/ipde/ipde/out/material-training/quality-review-2k-20261008/index.html`, with a packed Blender scene beside it. It contains 64 whole-material renders plus 12 native detail views of stucco and soil. In the fixed-strength displaced-photo/clay views, brick regains substantial mortar relief and stucco has stronger pitting. Brick remains somewhat irregular/soft; stucco's finest grain is still smoother than the reference; soil needs scrutiny for extra roughness. These are visual observations, not production approval or fresh-material generalization.

1. Open the 2K review, compare `starting_head` and `trained_2k` with `flat` and `target`, and record useful changes and misplaced relief. Save the edited review JSON. Judge at an appropriate shared artistic displacement scale; the default 3% width is an inspection setting, not measured material dimensions.
2. For historical reproduction only, resume the completed optimizer into a new run. This extends each crop from 100 to 200 updates:

   ```sh
   .venv/bin/python scripts/material_training_cycle.py resume --allow-retired-experiment \
     --resume-checkpoint out/material-training/native-2k-material-cycle-01/checkpoint.latest.pt \
     --updates-per-crop 200 --max-minutes 20 \
     --output out/material-training/native-2k-height-continued-02
   ```

3. To reproduce an archived roughness experiment, change `roughness` to `normal` and use a new output directory for independent normal supervision:

   ```sh
   .venv/bin/python scripts/material_training_cycle.py train --allow-retired-experiment \
     --dataset /opt/ipde/material-dataset/pilots/native-2k-four-material-01 \
     --target roughness --expected-size 2048 --updates-per-crop 100 \
     --selection final --prediction-limit 4 --max-minutes 20 \
     --output out/material-training/native-2k-roughness-01 --allow-unreviewed
   ```

All 582 normal/roughness labels across the full 97-material index passed CPU loading/loss preflight. Native 2K MPS forward/backward/export smoke runs also completed four actual updates for each target. Those smoke runs prove execution, not learned quality. Tiny UInt16 auxiliary-alpha differences are ignored independently without multiplying RGB/scalar values; meaningful transparency remains rejected. Short interpolated normal vectors receive directional supervision while their original encoded values remain available to the direct encoded-value loss.

## Precision and independent targets

Store the source maps at their original precision. All current height parents are UInt16 PNGs; conversion to Float32 happens only in active computation and does not invent additional source precision. Numeric height, roughness, and OpenGL normal maps use linear/Non-Color values in Blender. Diffuse uses its declared color transfer, normally sRGB. Metadata display tags on a numeric map do not authorize changing its stored values.

Train roughness on the actual roughness labels and independent normals on the actual normal-vector labels. Height-derived normals are useful for height review but do not contain every detail of the photographed material's separate normal map. Current normal labels include 92 UInt16 and 5 UInt8 materials; roughness includes 90 UInt16 and 7 UInt8. Keep those actual precision declarations instead of describing everything as 16-bit because the height map is.

## Review older frozen/LoRA candidates

```sh
.venv/bin/python scripts/export_material_candidate.py \
  --dataset /opt/ipde/material-dataset \
  --material white_stucco_02 --material farm_soil \
  --material cotton_jersey --material broken_brick_wall \
  --candidate frozen=out/material-training/four-material-adaptation-01/frozen/checkpoint.final.pt \
  --candidate lora_selected=out/material-training/four-material-adaptation-01/lora/checkpoint.selected.pt \
  --candidate lora_final=out/material-training/four-material-adaptation-01/lora/checkpoint.final.pt \
  --output out/material-training/my-visual-candidates --allow-unreviewed
```

This exports native Float32 EXRs and a review manifest, retaining checkpoint identities and steps. It uses the saved candidate's actual adapter state. It preserves the original diffuse/roughness maps and includes the original height reference. Choose a new output directory for each export.

Render a review with Blender installed:

```sh
.venv/bin/python scripts/review_material_quality.py \
  --manifest out/material-training/my-visual-candidates/review-manifest.json \
  --output out/material-training/my-material-review \
  --device METAL --preview-size 512 --samples 64
```

Use CPU or fewer samples for a quick check. The generated manifest, HTML, contact sheet, editable review JSON, portable Blender script and packed `.blend` make the comparison inspectable. Numeric shader assets use Float32 lossless EXR; the rendered PNG is a display preview. Render assets can be recreated from retained maps/checkpoints. Edit the manifest's shared displacement strength for an artistic comparison; do not normalize each height map separately to make it look stronger.

Ratings run from 0 (poor) to 5 (good). Noise and lighting scores rate cleanliness: a higher score means fewer unwanted artifacts. Choose `usable`, `needs_work`, or `reject` and describe conspicuous misplaced bumps/pits. The browser's save button downloads your edited JSON; keep it beside the review for the next cycle. There is no automatic metric winner.

For a roughness comparison, set the group `comparison_target` to `roughness`, provide one shared `height` map, and give each variant its own `roughness` path. That keeps geometry identical while you judge highlights and gloss. For directly trained normals, set `comparison_target` to `normal` and give each variant an OpenGL `normal` path; the review uses normal shading independently of height. Per-variant roughness is forbidden in height/normal comparisons to avoid changing two things at once.

To add native-detail views to an existing packed review without duplicating map assets, use `render_material_detail.py` inside Blender. It preserves the `.blend` on disk, geometry, lights and height strength, changes only the shared detail camera/render framing, and refuses to overwrite prior supplemental renders:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background \
  /opt/ipde/ipde/out/material-training/quality-review-2k-20261008/material-quality-review.blend \
  --python /opt/ipde/ipde/scripts/render_material_detail.py -- \
  --review-directory /opt/ipde/ipde/out/material-training/quality-review-2k-20261008 \
  --sample white_stucco_02_auto_003 --sample farm_soil_auto_003 \
  --variant target --variant starting_head --variant trained_2k \
  --preview-size 512 --samples 32 --device METAL
```

Those views already exist in the completed review. For a subsequent run, substitute that review's directory and scene/variant names. Tilted projection changes vertical sampling; the recorded camera footprint is an inspection aid, not proof of texture accuracy.

## Storage cleanup

The October 8 cleanup removed 696 explicitly listed generated files totaling 2,948,995,115 bytes. All retained checkpoints, JSON metadata, unbound unique raw baselines, current previews and pinned pretrained files were verified afterward. Original parent maps and prepared training crops were preserved. The cleanup plan/result record exact deleted hashes and retained regeneration paths; whole experiment directories were not removed.

New cycle exports default to compact Float32 EXR plus original PNG path/checksum references. Redundant NPY copies are optional with `--write-npy`. Keep checkpoints and the dataset/source recovery recipe before discarding generated preview assets.
