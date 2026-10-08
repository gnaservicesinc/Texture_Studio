# Native material crop and height-training workflow

This source tree contains the implementation and documentation. Experiment checkpoints, generated maps, source snapshots, recovery recipes and reports are retained locally and are not included. Record names below identify that local archive.

The preparation script operates on `/opt/ipde/material-dataset/sources/<material>/`. It identifies asset names from the downloaded map filenames, standardizes source folder names, and writes self-contained native 1024×1024 training crops under `samples/<asset>_auto_001/` and `_auto_002/`. The crops start at `(0, 0)` and `(width−1024, height−1024)`. A third, disjoint crop under `_auto_003/` supplies validation, preferring the center, then bottom-left or top-right. There is no parent resize. Existing manually edited stucco samples remain separate and are not selected by the generated dataset index.

From any directory:

```sh
bash /opt/ipde/ipde/scripts/prepare-materials.sh /opt/ipde/material-dataset
```

This uses the repository's existing `.venv`. Set `IPDE_MATERIAL_PYTHON` to another Python interpreter with NumPy and OpenCV installed if needed. The wrapper performs folder normalization, region preparation, and independent source-slice verification. It archives exact duplicate folders under `archived-duplicates/` with a restoration record; differing copies fail explicitly. All individual commands are available through `scripts/material_dataset.py --help`. Changed source payloads, altered generated crops, foreign indexes, and ambiguous paired dimensions fail explicitly; generated maps are not overwritten. Incomplete materials are reported and can be picked up by a later run.

The wrapper explicitly migrates the prior whole-material split to region validation. Exact previous indexes and sample metadata are preserved under `metadata-backups/` before publication. Existing crop PNGs remain unchanged; the migration adds only the missing regions and changes split metadata. Earlier training runs retain their original frozen dataset snapshots. An interrupted multi-file metadata update is detectable through index/sample disagreement, and the preserved backup supplies the original records for recovery.

An optional second argument `2048` prepares native 2K crops in `prepared-2048/`, preserving the existing 1K dataset. This is a separate preparation run, since 1K crops cannot recover their larger parent regions. Both resolutions retain their original sample precision. Where two training corners leave no room for disjoint validation, the script uses one training corner and the opposite corner for validation. A source too small for even those two disjoint crops is reported without resizing or overlap.

Each sample contains `diffuse.png`, `displacement.png`, `normal.png`, `roughness.png`, `sample.json`, and any additional paired downloaded PNG maps and text notes. Maps keep their real original integer sample depth. DirectX normals become OpenGL normals using the reversible integer operation `G = maximum_code − G`; that change is recorded separately from exact copies. Checksums cover the original file, the crop file, and decoded integer pixels. Required maps share one integer crop rectangle.

The current learning pilot uses every material in training. Validation uses a separate native region of each material, chosen to avoid overlap with its training crops. This checks whether the model learns the expected material detail before testing generalization on fresh materials. The older 36-material experiments used a whole-material split, and their frozen manifests remain with those runs. The trainer accepts cross-split material identities only when the dataset explicitly selects region validation and proves non-overlapping crop rectangles.

The initial October 7 refresh contained 168 native 1K samples from all 56 materials: 112 training corners and 56 disjoint validation regions. Every sample passed decoded source-pixel comparison. The independent migration audit confirmed 484 cross-split map comparisons, 476 unchanged existing cropped PNGs, and exact prior metadata backups. Published-file provenance is verified for 55 Poly Haven materials and Wicker013's AmbientCG photogrammetry package. See [the dated results](material-learning-validation-2026-10-07.md).

The earlier frozen batch contained **69 materials and 207 samples: 138 training and 69 validation**. It added 13 materials and 39 native crops. All prior 168 manifests, 726 crop PNGs and old index entries are unchanged; all 207 pass full decoded parent-region verification. The locally retained `source-refresh-69/` records preserve the audit, active index snapshot and new recreation recipe. Later source arrivals were recorded outside that historical batch for the subsequent continuation. Keep each source folder's matching diffuse, displacement, normal and roughness maps at their original precision; the scripts perform cropping and metadata generation.

The current frozen continuation contains **97 materials and 291 samples: 194 training and 97 validation**. It adds 28 complete materials and 84 crops, preserving all prior 207 manifests, 882 PNGs and index entries exactly. All 291 pass full decoded source-region verification; 830 all-role train/validation comparisons confirm disjoint regions. The new source and preparation proofs are retained in the local `source-refresh-97/` archive. The four-material adapter comparison used its original eight crops and did not train on these new sources.

The only 8-bit-displacement set, `brick_4`, was removed from source membership at the user's request and moved intact to macOS Trash. Other maps at 8-bit precision remain accepted. `leaves_forest_ground` is incomplete and stays untouched outside the index. AmbientCG Snow013/Snow014/Snow015 keep their original provider filenames and verified ZIP-member provenance, distinct from Poly Haven snow_01/snow_02.

## Numeric encoding and Blender exports

Displacement must have original 16-bit integer samples. Discovery and preparation exclude 8-bit height targets before publishing crops; 8-bit diffuse, normal or roughness sources remain accepted and accurately labeled. This check reports an unsuitable source and does not delete it automatically.

Native grayscale-plus-alpha PNGs retain both original integer components in their cropped files. The decoder proves that expanded RGB values are exact grayscale replicas, then restores the original two-channel layout; the dedicated encoder preserves PNG byte order. A declared scalar-alpha policy lets training use the original grayscale codes alone when UInt16 alpha is within eight codes of fully opaque. Alpha never scales the height target. Meaningful scalar transparency requires review, while RGB/RGBA scalar maps keep their existing stricter opacity checks. Explicit numeric-transfer overrides also leave auxiliary alpha unchanged.

Display color profiles on numeric texture PNGs do not authorize transforming their stored values. The source audit checked full-file MD5 and byte size against the official Poly Haven API, and complete SHA256/byte equality against verified AmbientCG ZIP members. An exact original-file match establishes unchanged downloaded bytes, rather than physical accuracy. Numeric height, normal and roughness codes are loaded directly without display gamma correction. Explicit transfer overrides exist for separately established non-linear numeric data; they are never inferred from a PNG gamma/profile tag alone.

GIMP's Encoding menu controls channel encoding and storage precision during processing. It does not establish the final exported file's numeric semantics. The manually edited stucco normal samples changed enough to match a power-gamma-to-sRGB re-encoding; fresh native parent crops avoid that conversion. [GIMP Encoding documentation](https://docs.gimp.org/3.0/en/gimp-image-encoding.html), [PNG color information](https://www.w3.org/TR/png-3/#11gAMA).

Finished app exports use 8-bit sRGB diffuse PNG and linear HALF/FLOAT EXR roughness, OpenGL normals and displacement. The generated Blender script marks diffuse as `sRGB` and all numeric maps as `Non-Color`. Training predictions use lossless ZIP-compressed FLOAT32 EXRs, checked against their Float32 arrays after decoding; previews are separate 8-bit files and are never used as numeric output. This pilot does not replace the app's selected model automatically.

## Training and inference

The first trainable model is a compact supervised material-height U-Net, operating on native 1024 inputs and targets. It is a texture-height experiment, not an adapted DA3 GIANT checkpoint. Diffuse is converted from its declared sRGB encoding to linear RGB in memory. Original UInt16 displacement codes become Float32 values divided by 65535 for the active sample. Targets are not gamma-corrected or independently stretched. Float32 height and multiscale finite-difference losses supervise both relief and detail. Normal and roughness maps are retained for future supervision; this trainer currently predicts height and calculates OpenGL normals from that height.

```sh
/opt/ipde/ipde/.venv/bin/python /opt/ipde/ipde/scripts/train_material_height.py train \
  --dataset /opt/ipde/material-dataset \
  --output /opt/ipde/ipde/out/material-training/my-height-run \
  --device mps --epochs 20 --max-steps 1200 --max-minutes 15 \
  --allow-unreviewed --offset-invariant-loss --gradient-weight 16 \
  --highpass-weight 4 --learning-rate 0.0005 --evaluate-training --prediction-limit 4 \
  --mask-transparent-input
```

`--allow-unreviewed` records the explicit experimental use of automatically prepared crops; it does not declare human quality approval. Training reports the first real forward/backward/optimizer step, changed parameters, sampled Metal allocations, checkpoints, validation metrics, Float32 predictions, and previews. `--evaluate-training` separately measures fit on all training crops. Metrics cover every eligible crop; `--prediction-limit` bounds exported examples. Run directories must be new. A successful training step establishes that this architecture fits, rather than establishing GIANT LoRA capacity or production quality.

Meaningful diffuse transparency is rejected by default. `--mask-transparent-input` keeps straight RGB and numeric targets unchanged, excluding nonopaque pixels plus an eight-pixel margin from loss and metrics. Gradient pairs, pooled blocks, high-pass neighborhoods and offset means use only eligible pixels; the margin does not guarantee that all network feature influence from those input pixels vanishes. Masked prediction exports include a separate validity array and record the evaluated fraction. The current wooden-floor center crop retains 98.13% eligible pixels after this margin; all 112 training crops pass without meaningful transparency. This mode never composites the image, invents target values, or changes source files.

```sh
/opt/ipde/ipde/.venv/bin/python /opt/ipde/ipde/scripts/train_material_height.py predict \
  --checkpoint /opt/ipde/ipde/out/material-training/my-height-run/checkpoint.best.pt \
  --input /path/to/photo.png --input-encoding srgb \
  --output /path/to/new-prediction-directory --device mps
```

Evaluate held-out height error, native/multiscale gradient error, and recovered detail against untrained, constant and luminance baselines. `scripts/evaluate_da3_material.py` additionally preserves pinned GIANT's raw depth and measures an explicitly optimistic ground-truth scale/offset alignment. That diagnostic uses the target to align the unrelated relative-depth units, so its aligned result cannot be treated as an exportable prediction for a new photo.

The source crops are high-quality material maps, but a small reflectance-to-height pilot is not evidence that casually photographed, shadowed surfaces will generalize. View the held-out relief and normals before promotion. A lower scalar error with flattened fine detail is not a successful detail model.

The October 7 native-size probe measured the same 218,209-parameter checkpoint, Float32 optimizer and losses, batch one, one warmup and five measured steps at each size. Native 1K averaged 0.160 seconds per training step with 2.19 GB sampled Metal driver allocation; native 2K averaged 0.598 seconds with 8.18 GB. Thus 2K is practical for this compact model on the local 64 GB Mac, approximately 3.75 times slower in the measured compute steps. Image decoding, validation and output writing add time. This does not establish GIANT LoRA capacity or improved 2K quality. The probe retained original source SHA-256 and crop coordinates without creating a duplicate 2K dataset.

Two 600-step native-1K pilots completed on the original 36-material snapshot. The second used mean-offset-invariant height loss, native multiscale gradients and radius-4 detail loss. Mean removal occurs only inside the loss; the stored target values and amplitude remain untouched. Prediction can optionally add a recorded constant neutral height origin, without rescaling or clipping. On eight validation crops from four independent materials, this pilot reduced native gradient error by about 2.6% and high-pass error by about 3.3% against flat height, but retained only about 12% of target gradient energy. The paired DA3 comparison, even with target-based optimistic scale/offset alignment, also lacked fine relief. Both outcomes remain experiments; no model was automatically promoted into the app. See `out/material-training/quality-comparison.json`, the run summaries and contact sheets for the full evidence.

At roughly 56 independent source materials, a focused learning pilot is reasonable. Multiple regions and augmentation increase observations while keeping each source's validation region separate. These region checks measure learning on familiar materials. Fresh textures will be needed to measure behavior on new materials and casual-photo lighting later.

Before expanding the dataset, `scripts/diagnose_material_fit.py` tests whether the same compact model can learn one repeated, unaugmented native crop. The October 7 stucco and soil comparisons each ran 300 optimizer updates per objective from the same initialization, at learning rate 0.001. Both fitted overall training relief with about 0.99 correlation. A target-energy-weighted squared objective improved fine-detail correlation on both materials. This establishes capacity on these repeated crops, not generalization or adequate fitting across all 56 materials. See [the fitting experiments](material-fitting-2026-10-07.md).

```sh
/opt/ipde/ipde/.venv/bin/python /opt/ipde/ipde/scripts/diagnose_material_fit.py \
  --dataset /opt/ipde/material-dataset --material white_stucco_02 \
  --output /opt/ipde/ipde/out/material-training/my-stucco-diagnostic \
  --steps 300 --learning-rate 0.001 --allow-unreviewed --mask-transparent-input
```

The whole-dataset trainer now offers `--objective relative-squared`. It divides squared height, multiscale-gradient and high-pass errors by detached reference-energy constants, with fixed numerical floors. This changes loss units, not stored values or the amplitude of the target. `--initial-checkpoint PATH` loads compatible weights into a fresh optimizer and records their hash; it is a warm start, not an optimizer resume. `--validation-every-epochs 5` reduces full-dataset evaluation overhead while preserving initial and final evaluation. Best selection uses the selected objective; current code retains the validated starting model if updates make it worse and snapshots the implementation sources for future runs.

The all-material region run completed 1,200 native-1K optimizer steps in 404 seconds. It reduced native gradient error against flat height by 1.38% on training crops and 1.25% on validation regions. Mean per-sample recovered gradient-energy ratios were 0.156 and 0.143 respectively. The visual comparisons show weak weave/seam structure but largely missing soil and stucco relief. This model still underfits its training material detail; improving fitting is the next task before evaluating fresh-material generalization. The checkpoint and full reports are retained under `/opt/ipde/material-dataset/experiments/native-height-region-pilot-02/` and were not promoted into the app.

## Keeping or removing parents

Generated crops and metadata are self-contained: training does not require the 4K parent images. The source parents remain in place for this fitting work. Two corner crops do not preserve the rest of the material, so recreating larger or different regions requires downloading parents again. Preparation never deletes sources. Bi Stretch was moved to macOS Trash separately at the user's request because the paired dimensions differed; it was not part of the generated dataset.

`scripts/material_recreation.py export` records source identities, verified download evidence, original precision, crop coordinates, encoding/normal transforms, exact sample metadata, source notes and the preparation code itself. It does not copy image arrays. The compressed recipe is `/opt/ipde/material-dataset/recreation.json.gz`; `plan --recipe PATH` lists required parents and unresolved origins without downloading anything. Re-downloaded files must pass the recorded full SHA256 and byte count before recropping; published assets can change. Decoded crop hashes establish sample equality even if a different PNG encoder changes container bytes. The recipe covers generated indexed samples, rather than separately edited manual crops.

The preserved 56-material, 16.55 MB compressed recipe identifies all 242 selected parent files: 238 direct Poly Haven downloads and four members of one verified AmbientCG archive. Wicker013's filenames were changed locally, but each file matches its original archive member exactly. Its source page identifies Surface Photogrammetry; this does not imply that all AmbientCG assets are photographed. Source notes include a large original text file, accounting for most of the recipe size. The archive verification download was removed after auditing; source parents and training crops remain in place.

`/opt/ipde/material-dataset/recreation-69-materials.json.gz` separately records the historical 69-material, 207-sample batch and all 294 required parents (290 direct Poly Haven files plus four Wicker archive members). Its size is 16,588,251 bytes and full SHA256 is `56932c5ba771473ec2f37c3b92ed437c496a5ac2624993e1ef83c87f91680f8b`. No required origin is unresolved. This new recipe leaves the historical `recreation.json.gz` untouched.

Published 2K originals are a separate scale experiment. Use the provider's files rather than locally resized 4K maps, as the user found better detail in the published versions. A 1024 crop from a 2K parent covers a larger fraction of the surface than a 1024 crop from its 4K parent. Split checks therefore need parent-normalized surface footprints across resolutions: different filenames or pixel coordinates alone do not establish disjoint validation. Do not append these examples to the current region index without that check.

`scripts/download_material_resolution.py` plans or downloads the provider's original paired PNGs, checking published MD5/byte counts and actual PNG headers. The current pilot has downloaded eight native 2048×2048 UInt16 PNGs for `farm_soil` and `white_stucco_02` under `sources-2k/`, consuming 125,341,362 bytes. A repeated run reused every verified file without another image download. The metadata-only plan for all 55 eligible Poly Haven materials totals 220 PNGs and 3,139,853,183 bytes; that full set has not been downloaded. Height requires original UInt16, while actual 8-bit normal/roughness maps remain accepted and labeled at their real precision. No upconversion or local resizing occurs.

```sh
/opt/ipde/ipde/.venv/bin/python /opt/ipde/ipde/scripts/download_material_resolution.py \
  --dataset /opt/ipde/material-dataset --resolution 2k --plan \
  --report /opt/ipde/ipde/out/material-training/my-published-2k-plan.json
```

For a bounded download, add `--materials farm_soil --materials white_stucco_02 --download` in place of `--plan` and use a new report filename. The source records preserve URLs, provider/license, resolution, full checksums, verified native dimensions and original bit depths. This downloader intentionally does not assign training splits or change the active dataset. The user has compared Poly Haven's resizing quality; AmbientCG's alternate-resolution quality has not been established.

Eight additional provider-original 2K maps supplied with `bamboo_wall_02` and `bamboo_wall_03` are now kept separately under `sources-2k/`, preserving their full original bytes. Check source footprints before assigning any scale crops:

```sh
.venv/bin/python scripts/plan_material_scales.py \
  --dataset /opt/ipde/material-dataset \
  --parents /opt/ipde/material-dataset/sources-2k \
  --output out/material-training/published-2k-footprints-next.json
```

The completed four-material plan found no geometry-disjoint candidate among each 2K parent's 1K corners and center, because the current 4K center validation region straddles them. Actual registration remains unverified. Use a separate versioned cross-resolution split for a scale experiment; do not relabel or overwrite existing validation crops.


The current compact recovery recipe is `/opt/ipde/material-dataset/recreation-97-materials-parent-note-refs.json.gz`, also preserved in the local `source-refresh-97/` archive. It is **319,073 bytes**, with SHA256 `f84eb5f1f5d2e178dbbc92f6953471f62c451071aa33a19b8f13b8b64962005a`. All 415 required parents have verified download origins, using 403 distinct downloads. The two PNG files named `.txt` are restored from exact parent references; the unique 492-byte README remains embedded unchanged. A real local restoration matched every original note byte and a second restoration was idempotent. The earlier full 97-material recipe and historical 56-/69-material recipes remain unchanged.

After obtaining and verifying parent maps in a separate source directory, restore source notes before regenerating the recorded crops:

```sh
.venv/bin/python scripts/material_recreation.py restore-notes \
  --recipe /opt/ipde/material-dataset/recreation-97-materials-parent-note-refs.json.gz \
  --sources /path/to/recovered/sources
```

The command checks parent hashes/sizes and existing destinations, preserves original filenames and refuses to overwrite differing files. It uses no network and also accepts older recipes with embedded source notes. Full source-map downloading and crop rebuilding remain the explicit recovery steps in the recipe.
