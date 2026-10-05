# IPDE Studio

For the current 0.9 training controls, see the bundled
[user manual](manual/index.html) and [interactive guide](help/index.html).
Development save and dataset formats may change before 1.0.0.

Launch `make studio` (or `build/IPDE Studio.app`). Open/create a project, then
choose **Effect / map**, **Depth estimation**, **Photo effects / masking**, or
**Manual**. Guided modes hide technical controls; **Advanced settings** exposes
individual overrides. Manual exposes all choices. Each project stores its
workspace and `project.ini`; datasets and models are not silently selected into
another project's extraction settings.

Studio launches **Extract maps**, **Manage & review datasets**, and **Train model &
compare**. Dataset Studio owns every dataset edit, including optional training-set
preparation, compacting, archiving and links. Trainer reads datasets and manages
training runs and models. A request for another step opens or focuses the correct
project app and selects the requested dataset.
The hub locks each app/project combination, broadcasts filesystem changes, and
keeps background tasks alive when its window is closed. Apps for different
projects can run concurrently. Direct GUI subapp launches require a Studio
session; `--smoke-test` is the validation exception. CLI tools remain standalone.

## Dataset and review workflow

Name datasets after the things you photograph; use **Category** to label subject
types. Groups are optional. A subject type such as flowers is not automatically
an independent scene: use scene groups only for related captures that must stay
together during validation. Metadata columns and search expose phone model,
capture date/time and source filenames.

Dataset Studio keeps the dataset library in a left sidebar. Select any row and
open **Photos and depth**; role assignments on **Training split** are separate
choices for combining datasets. **Dataset actions** holds inspection, storage,
archive, removal and linking commands. Membership and split changes save to the
selected dataset automatically. Durable pending drafts survive switching rows
and reopening Studio; save failures keep them available for retry.

In **Photos and depth**, each photo's **DepthPro**, **DA3** and **V2** buttons
generate a missing result immediately from stored full display RGB. Select several
photos for **Enable teacher for selected photos…**. Turn an enabled teacher off
to discard unneeded computed maps; enabling it later regenerates them. Shared
files and source/reference arrays are protected. **Remove selected** also discards supported generated display teachers; legacy
or imported targets that these controls cannot regenerate retain their files. Generation runs all photos through one resident model
before releasing it and loading the next; matching DepthPro teacher/anchor
settings reuse predictions.

**Scan spatial directory** recurses without following links. Nonspatial photos,
portraits without calibrated stereo, malformed containers and inconsistent views
are skipped with a reason. Apple camera metadata and supported stereo structure
are required. These checks cannot establish authenticity of convincingly forged
metadata or prove an image was never AI edited.
Choose the folder containing original HEIC/HEIF/HIF photos, rather than the
project's workspace or generated dataset folder. Validated photos appear in the
list as the scan runs; cancelling keeps the photos already found. Scanning and
manual addition share the same duplicate handling.

Use the teacher buttons to turn one to three teachers **On**. Each teacher creates
an independently reviewable entry for a photo, sharing its raw data and validation
split. Select a poor teacher result and choose **Remove selected** without dropping
the photo's other results. Generation publishes
completed entries as it runs; review and preview jobs use a separate background
process. Interim split assignments remain provisional and cannot be trained,
composed or curated until generation finishes. A bad photo/result is recorded
and skipped; a filesystem failure stops safely.

Hover for a synchronized **1:1 magnifier**, or click for a floating pixel-size
preview centered on the clicked image point. Drag to pan; right-click, click
outside, or press Escape to dismiss. **Compare teachers** offers
shared meter contrast, overlays, disagreement and a draggable lit relief view.
Relative outputs use explicitly separate visual scaling; their differences do
not represent meters. Relief is a visual inspection aid, not calibrated 3D
reconstruction. Duplicate aliases for an identical target are hidden from the
view selector. Units and display ranges explain why some normalized views can
look alike despite different scientific values.

RAFT Studio opens **Train model & compare**. Select an existing dataset and
click **Start model training** to run epochs and save weights; a single dataset
does not need a preparation copy.

The optional **Training split** page in Dataset Studio accepts many datasets. Choose a dedicated
validation dataset, seeded random selection across all datasets, or equal random
counts of independent groups per dataset. Group preservation/override enables
A/B experiments; exact duplicates, teacher variants and known bursts always stay
together. External validation overlapping training sources is rejected. Use
**Compare selected model with baseline** on a spatial photo to inspect the
original and trained RAFT results with the same grid and meter contrast.
For an automatic split, set **Validation size** from 0% to 100% (20% by default). It counts
independent capture components, rather than individual teacher entries.
**Advanced split options** exposes the repeatable seed and authored-group
preservation. **Set split…** in **Photos and depth** instead controls individual
capture assignments directly in the selected dataset. **Apply split and open
selected dataset in Trainer** saves the selected automatic validation percentage to that dataset
before opening it. At 100%, every included component is validation; Trainer
reports that no training examples remain.
At 0%, no validation examples remain and training also requires a different split.
Select a dataset row and choose **Use for training** to include it. The **Use**
column displays **Training**, **Validation**, or **Not used**. **Use for validation**
holds out the entire selected dataset and switches to the dedicated validation
strategy; **Do not use** removes it from the set. Selecting a row chooses the
active dataset without changing its assigned role. With a single available dataset,
its role starts as **Training**. Roles stay assigned when you refresh the library.
Switching to an automatic validation strategy keeps existing **Validation** roles
visible; assign those datasets to **Training** or **Not used** before creating
the set. The message above **Create training set and continue** explains missing selections,
name conflicts, pending photo/split saves, or an active task. Wait for edits to
finish saving before creating a combined training set.
Creating a set verifies the selected arrays and reuses their existing storage.
On supported macOS filesystems it creates independent copy-on-write clones;
otherwise it uses immutable hard links on the same filesystem. It does not
recompress or make another full array copy. Verification can still take several
minutes for large datasets.
The button reads **Creating training set…** during this work; the progress
message reports reused arrays and final verification. Completion selects the saved
set and opens Trainer at the model training step. Preparing a set assigns validation splits;
**Start model training** runs the optimizer and writes a checkpoint.

Studio windows fit within the screen's usable area. Scroll to reach controls
on smaller displays; review captions have their own space beneath the images.

## Storage and external datasets

New GUI datasets use single-channel lossless ZIP EXR for each float32 teacher
result, exact PNG for unsigned RGB, and NPZ for masks/other layouts, with bit-exact
round-trip checks. Identical arrays are stored once even when several records or
teacher variants use them. Raw arrays are never normalized or gamma corrected.
Native teacher predictions, confidence and unrelated auxiliary planes are omitted
by default; Advanced can retain them explicitly. Equivalent positive-finite masks
are derived from depth, while restrictive masks stay stored. Source photos stay
unchanged. The detail preset leaves metric anchoring off because relative labels
can train the display depth model directly; extra teacher and scale maps add storage.
Prepared training sets retain the original scientific files byte-for-byte through
shared storage. Their array paths stay inside the new set, so removing the source
directory does not break the set. Hard-linked payloads must remain immutable;
Studio's array writers publish new files rather than editing existing ones.
The storage column reports reused arrays and newly added metadata. Sources and
prepared sets must be on the same filesystem; train a linked dataset directly
when its files are on another volume. The CLI's explicit
`compose-datasets --storage-mode copy` creates a portable compressed copy when
requested. Existing older copies are left in place.
**Compact dataset** creates a new verified EXR/PNG copy of an older dataset;
it preserves every existing scientific plane and split assignment and leaves the original available. Using both
copies consumes additional storage until you remove the original yourself.

**Remove generated dataset…** in Dataset Studio permanently removes an owned
generated dataset after a confirmation dialog. Original source photographs and
trained models remain. Linked external datasets cannot be cleaned up here.
Prepared sets with contained clone/hardlink files survive source removal; shared
blocks are only released after their last owner is removed. Cleanup refuses
source/model files placed inside the dataset, external array dependencies, and
datasets being read by an active CLI task. **Clean run files…** in
Trainer retains all output checkpoints and their provenance reports. Removing
dataset storage is a separate explicit Dataset Studio action. Neither action is
automatic, and neither uploads data.

Set **File processing threads** in the project's Studio window to override concurrency.
Automatic uses available cores. Verification deduplicates repeated file records
and checks independent files in parallel; composition, compacting, reviewed
copies and training target preparation use bounded worker queues. The CLI
equivalent is `--workers 4` (or `--workers 0` for automatic). These file tasks
have no MPS filesystem interface. Teacher inference remains serialized to bound
GPU memory use; training uses MPS when **Device** is `auto` on a supported Mac
or explicitly `mps`, preserving FP32 rather than adding mixed precision.

New GUI datasets run teachers and optional metric anchors only on the full
display RGB photo. Full-display targets and native model predictions are retained
for review/export; they can add hundreds of MiB per photo. They are not
automatically warped or selected for stock RAFT training. The original display
RGB remains lossless. An explicit native-stereo experiment is a different output
task and needs compatible left-grid labels.
Training uses a bounded cache of decoded samples instead of loading the entire dataset into RAM.
NPZ reads decompress a requested plane into memory; NPY remains available with
`dataset --uncompressed` when memory mapping is preferable.

Optional Hugging Face imports use the `datasets` extra with explicit column
mapping, NPY/NPZ planes, meter depth and compatible stereo calibration. Generic
image or text datasets cannot directly supervise calibrated RAFT. See
[dataset library](dataset-library.md) for mappings and examples; the backend uses
[data-only dataset loading](https://github.com/huggingface/datasets/blob/main/src/datasets/load.py).

## Model and precision background

Dataset Studio is the Qt app for creating and managing spatial-photo datasets.
Trainer learns full display teacher targets with an RAFT stereo-to-display depth model, or runs an explicitly selected stock RAFT native-left experiment, and exports checkpoints.
IPDE remains the extraction app: choose an exported checkpoint in its existing
**RAFT model** field and continue extracting images and depth maps.

Launch the installed local build:

```sh
make studio
# Or open build/IPDE Studio.app
```

Each project uses its `workspace` folder; the legacy standalone CLI workspace can remain `/opt/ipde/raft-workspace`. Its `datasets`, `runs`, and
`exports` directories contain ordinary local files. Dataset creation and training
run in a background Python process; progress appears in the app and a task can
be cancelled. Model inference never uploads photographs or downloads weights.
See [model setup](model-setup.md) for explicit, revision-pinned downloads.

## Create a dataset

Add spatial HEICs in **Add photos / new dataset**. A **teacher** is the depth model
that generates the target maps RAFT will learn to imitate. Its targets are
estimates: a detailed-looking map can still contain incorrect surfaces or
distances. Review the maps before training.

Edit the group column so related captures of one scene have the same group.
For example, several photos of the same room or object during one capture
session belong together, even if the camera moves. Keep repeat shots and
variants together; each HEIC's left/right views already stay together. Use
different group labels for scenes that really are independent.

**Independent scene groups** records your
confirmation that you have grouped the photos this way. It lets the report
describe the held-out groups as scenes. The app does not automatically verify
scene identity, and turning the option **On** does not verify depth-map quality.
Do not enable it just because photos have different filenames or group labels.

Grouping prevents **validation leakage**: if RAFT trains on one photo of a room
and is tested on another very similar photo, its score may look good without
showing that it works on a new scene. Metadata catches exact source/RGB
duplicates and reported bursts, but cannot reliably recognize the same room or
object. Keep this option **Off** when scene independence is uncertain; the
report then uses the default capture grouping. Supplied group labels still
keep related captures in the same split when the option is **Off**.

Choose the teacher and its local checkpoint/source directories:

- **DepthPro** provides estimated meter depth and is the initial RAFT teacher.
- **Depth Anything V2 Large** provides relative inverse depth. Its processing
  size is configurable; the GUI begins at native input (zero).
- **Depth Anything 3 GIANT 1.1** provides relative depth in the current adapter.
  The app does not claim that a larger model guarantees better geometry.

**Depth scale — Estimate meters with a second model (DepthPro)** converts a
relative teacher's values into an
estimated meter scale using DepthPro. V2 and the current DA3 adapter describe
near/far relationships, but their values have an image-specific scale: a value
of 2 does not mean 2 meters or necessarily the same distance in two photos.
The anchor runs DepthPro on the same full display RGB image and fits a scale and offset in
inverse depth. It creates a separate estimated-meter map from the relative
prediction; the original teacher values remain unchanged.

Use the anchor when you need an estimated meter-scale display map from **V2 or
DA3**. The option is disabled for **DepthPro**, which already estimates meters.
You can leave it off when creating relative maps for visual review, effects or
teacher comparisons. Meter scale alone does not align different camera grids or
make a display target suitable for the unmodified RAFT decoder. Independently
measured meter-depth references aligned to the native left grid remain a
separate supervised-training alternative.

The anchor is therefore useful for a specific purpose, not something to always
enable. It adds another model inference and requires DepthPro's local weights
and source. Its scale can be inaccurate because DepthPro is also an estimate;
agreement between two models is not a distance measurement. A fit with too
little depth variation or poor agreement is rejected. Review the actual
training target for geometry that looks wrong. Training automatically skips
targets whose anchor was rejected, without changing their relative values or
substituting another stored teacher. Anchoring cannot repair hallucinated objects
or surfaces.

**Input size** controls a configurable teacher's effective processing resolution,
not the full reference-photo dimensions or saved output size. Teachers receive
the complete display photo. V2 uses the configured shortest side; DA3 uses the
longest side. The GUI begins at zero, requesting native source size with
14-pixel divisibility processing. The field also applies to a configurable
secondary teacher. DepthPro always resizes internally to 1536×1536 and uses its
fixed pyramid/decoder, regardless of this configurable-teacher field.
Requested and actual processing dimensions are recorded. DA3's current adapter
evaluates single views independently. Native GIANT processing at 5712×4284 on
64 GB memory is unverified and can require very large working arrays. Compare
reviewed results and consult the [manual's memory discussion](manual/index.html#full-display-teacher-memory)
before assuming native processing fits.

**Inference device** chooses where the model runs. **auto** selects an available
accelerator, falling back to CPU; **mps** uses supported Apple GPU hardware,
**cuda** uses a supported NVIDIA GPU, and **cpu** uses the processor. Choose a
device your machine
supports; CPU can be useful when accelerator memory is insufficient, though it
can be slower. Device choice does not certify depth accuracy.

Datasets preserve all decoded RGB/auxiliary arrays as exact NPY or lossless NPZ files, camera
calibration, source/model hashes, source revisions, native prediction grids,
source-grid float32 predictions, masks, and grouping/split metadata. Depth
targets never pass through an 8-bit preview. Normalized RGB tensors used by a
model are separate from the saved raw data. Training checks array hashes before
loading targets.

## Manage photos and review depth maps

Generation opens **Photos and depth** when it finishes. Select any existing dataset
in the sidebar to inspect its photos and teacher targets. Select a photo or a
teacher row to see its RGB image beside its generated depth map.
Click either image to open a native-pixel preview centered on that point. Drag
to pan, and right-click or click outside to dismiss it. Hover for a synchronized
1:1 magnifier.
Look for incorrect object boundaries, flattened or invented surfaces, holes,
and inconsistent near/far order. Select multiple photo rows and choose
**Remove selected** to exclude their targets from the selected dataset;
**Restore selected** includes them again. Select an individual teacher row and
choose **Remove selected** to remove just that result while keeping other teachers
for the same photo. **Remove and next**
supports reviewing one photo or teacher target at a time. Removed rows remain
visible with a strike-through. The **Status** column reads **Included**,
**Removed**, or **Some removed** when a photo retains only some teacher targets.
Removing a target excludes its complete training
sample, including both stereo views.

Use **Set split…** to move selected entries to training or validation. The change
also moves their entire known duplicate/photo/teacher/burst/scene component,
including related rows outside a filter. The totals show included targets and
save state immediately. Changes save automatically without copying or decoding
existing arrays. Removed rows remain restorable when you reopen the dataset.

**Add photos…** saves the new spatial HEIC paths in the selected dataset before
opening generation. The pending photo list survives reopening, so you can resume
generation from **Photos and depth**. Generate their depth targets to add them to
the selected dataset; existing photos and teacher results are not regenerated.
**Add from dataset…** includes active photos and teacher targets from another
prepared dataset using existing array storage, without
running inference. Any conflict between existing split assignments for the same
capture must be resolved by assigning its component one common split.

New default datasets retain display teacher/anchor maps for review and export;
they do not automatically supply stock RAFT labels. Check each map beside the
RGB reference from its own grid. In an explicit native-stereo experiment,
**Selected training target** shows the compatible map that distillation would
use. The label selector exposes available raw, native-model and anchored maps
so you can inspect the source of a problem. Viewing another label does not
change which target training uses. A visually plausible map is not enough to
establish its calibration or its compatibility with a model's output grid.

The grayscale depth preview uses **near = white, far = black**. **Magenta** marks
invalid or unsupported values. Each preview uses its own display range, so
matching shades in different photos do not imply matching distances. The
preview is a visualization only: the original float arrays, masks, and raw
RGB/auxiliary data are untouched. A visually plausible map still needs
independent accuracy checks for displacement work.

Membership and split edits atomically update the selected dataset's manifest.
Original scientific values and files stay unchanged, and existing teachers are not
rerun. The save status shows when changes are pending, saving, or saved. A
durable per-dataset draft preserves unfinished changes after a save failure or
reopening. Closing while a save is unfinished requires an explicit choice, and
opening Trainer waits for the current edits and chosen split to save.
Keep usable independent groups in both splits. If validation becomes
empty, set aside a suitable independent capture group using **Set split…**, or
use **Training split** to create a new automatic split. See
[dataset library](dataset-library.md) for the `update-dataset` CLI and edit JSON.

## Display image and stereo alignment

The HEIC display image can come from either stereo camera and has different
resolution, framing, and processing. RAFT's native correspondence reference is
the decoded left view. Scaling all three RGB images to the same size and
overlaying them is a useful alignment diagnostic; it is not a depth registration.

Dataset Studio's default teachers and metric anchors use full display RGB only,
and retain the full display grid plus each model's native prediction. They do
not fall back to a stereo-left teacher or automatically choose a registered
display-to-left training target. Camera viewpoint differences can cause
distance-dependent alignment; a single fixed warp cannot generally correct it.
Existing registration diagnostics are approximate and separately identified.
They do not change the default full-display teacher workflow.

Stock RAFT-Stereo's existing decoder predicts on its left input grid. Full display
teacher maps can be useful for direct effects/export/comparison, but cannot by
themselves train that decoder to produce a separate full display grid from native
stereo inputs. The default **RAFT depth on the display grid** model adds stereo encoders
and a learned query decoder. It receives whole native left/right RGB and predicts one map on the full display grid. Right-view content is correspondence-aligned to the left reference before fusion; one shared query decoder replaces the previous independent camera queries. Training uses the reference depth map as labels and compares every supported
output pixel with the original display teacher depth; no fixed camera
warp or stereo-teacher fallback is imposed. Its new head needs training, and
full-sized output is not a guarantee of accurate fine detail. See the bundled
[architecture note](manual/index.html#raft-stereo-to-display-depth-model).
The following native-stereo target formula applies only to
explicitly compatible experiments with aligned metric labels.

For a positive metric left-grid target, training uses:

```text
geometric_disparity = focal_pixels * baseline_meters / teacher_depth_meters
signed_flow = (cx_right - cx_left) - geometric_disparity
```

The presentation-only HEIC disparity adjustment does not enter this formula.
Out-of-image/crop correspondences and estimated occlusions are excluded.
Optional photometric support gives a stricter training mask. A mask does not
turn an estimated label into independently measured depth.

Review reports include display and stereo dimensions, the metadata's camera
identity and held-out registration errors. The investigation report also records
supported cells and capture warnings.
Equal output dimensions alone do not establish equal camera grids. See the
read-only [IMG_1689 investigation](alignment-and-checkpoints-2026-10-04.md).

## Train and export

Select a dataset and the visible **Model output** first. The default display
model uses full-display labels, whole native stereo inputs and one unchanged
unit convention per run. The stock RAFT picker instead requires calibrated
native-left labels. Eligibility, counts, Total Steps and error units follow both
the chosen model output and label mode. Configure the initial RAFT checkpoint/source
and select **Start model training**.

Display limited scope trains its added encoders/query decoder with RAFT frozen.
Full scope adapts RAFT too. Automatic display full scope additionally needs at
least 500 entries, medium/high quality and recorded native inputs no larger than
512 × 512; large or unknown input dimensions keep RAFT frozen. Stock limited
scope changes its refinement block, with optional full fine-tuning. Large native
full-network display training remains an explicit memory experiment.

An epoch visits every eligible image/teacher entry once. The display depth model
uses every supported positive finite full-display target pixel in decoder tiles;
stock RAFT draws native-resolution crops. **Steps Per Update** accumulates image
gradients before an optimizer update. Partial groups flush at epoch end. With T
entries and accumulation A, an epoch has `ceil(T / A)` optimizer updates.
**Total Steps** counts these updates, not tiles. Choose complete epochs or an
exact update budget; the latter can stop partway through an epoch.

**Depth tile side length (pixels)** controls temporary query-decoder memory/dispatch
overhead without resizing inputs or targets. The value 768 means at most 768 × 768 pixels per tile. For a 5712 × 4284 map, 8 columns × 6 rows = 48 portions cover one map, with smaller edge tiles; the output is still 5712 × 4284. In stock mode, **Native stereo
patch pixels** specifies original-pixel crop size, a multiple of 32. **RAFT
iterations** controls correspondence refinement work. Simple Quality adjusts
256/512/768 pixel tiles (stock crops), 16/24/32 added feature channels and
8/16/24 iterations. Length sets a capped dataset-aware duration and error goal.
Initial display learning rate is `1e-4`; stock is `1e-5`.

Validation can run each epoch or with saved checkpoints. Zero validation samples
uses the whole set; N chooses a fresh random subset capped at the eligible validation count. The default of up to 16 uses all three when only three held-out entries exist. The plan follows manual changes to iterations, tile size and validation settings immediately. Display scores are mean
`abs(prediction-target)/target`, a fraction (0.10 = 10% average relative error),
plus raw MAE in declared meters/relative/inverse units. Stock scores are flow
MAE in pixels. A passing subsample early-stop goal requires full-set confirmation.
Do not compare different unit conventions or camera grids pointwise.

Save intermediates each epoch, every N epochs/steps or only at the end. **Save
checkpoint now** queues a safe update-boundary save, available for export while
training continues. **Stop** saves recoverable training state. Resume into a new
destination with the original manifest and compatible architecture/units, mode,
scope, tile/crop size, iterations, accumulation, learning rate, seed and support.
The safe-error guard stops nonfinite, zero or excessive fractional display error
(stock: normalized pixel flow error), records why, and saves finite recoverable
state. This state is not automatically a good model.

Training does not modify dataset arrays. Native datasets receive metadata/split
checks first, then checksum/geometry checks as arrays are consumed. External or
unknown datasets receive full preflight; corruption aborts. Distillation learns
teacher estimates and their mistakes. Supervised references must be measured
and declared on the selected model's output grid; mixed uses eligible kinds.
No mode name or low teacher-agreement score proves physical accuracy.

Display checkpoints use `ipde-display-depth-v1`, reports
`ipde-display-training-report-v1`, and exported inference models
`display-model.pth`. Selecting one enables direct display prediction products;
stock RAFT correspondence/depth products remain separate. Compare independent
full images, each with its own RGB reference, and explicitly choose the model.
Keep direct display AI as a separate practical comparison.

```sh
.venv/bin/python raft_studio.py --json workspace /opt/ipde/raft-workspace
.venv/bin/python raft_studio.py dataset photo-a.HEIC photo-b.HEIC \
  --output-dir /opt/ipde/raft-workspace/datasets/my-device \
  --model depthpro --groups scene-groups.json
.venv/bin/python raft_studio.py dataset photo-a.HEIC photo-b.HEIC \
  --output-dir /opt/ipde/raft-workspace/datasets/my-device-v2 \
  --model depth-anything-v2 --input-size 0 --metric-anchor depthpro \
  --groups scene-groups.json
.venv/bin/python raft_studio.py review-dataset \
  /opt/ipde/raft-workspace/datasets/my-device
.venv/bin/python raft_studio.py preview-sample \
  /opt/ipde/raft-workspace/datasets/my-device \
  --sample SAMPLE_ID --label training --output-dir /opt/ipde/raft-workspace/previews
.venv/bin/python raft_studio.py curate-dataset \
  /opt/ipde/raft-workspace/datasets/my-device \
  --keep SAMPLE_ID --keep ANOTHER_ID \
  --output-dir /opt/ipde/raft-workspace/datasets/my-device-reviewed
.venv/bin/python raft_studio.py train /opt/ipde/raft-workspace/datasets/my-device \
  --checkpoint /opt/ipde/raft-workspace/runs/my-device/checkpoint.pth \
  --raft-root /opt/ipde/RAFT-Stereo \
  --raft-model /opt/ipde/models/raftstereo-middlebury.pth \
  --student display --limit-mode epochs --epochs 10 --steps-per-update 1 \
  --patch-size 512 --iterations 16 --device mps \
  --validation-schedule checkpoint --validation-samples 16 \
  --checkpoint-schedule epochs --checkpoint-every 2
.venv/bin/python raft_studio.py export \
  /opt/ipde/raft-workspace/runs/my-device/checkpoint.pth \
  --output /opt/ipde/raft-workspace/exports/my-device
```

`scene-groups.json` maps absolute source paths to group labels. Use `--grouping
scene` only when those labels actually describe independent scenes. The default
is capture grouping. For teacher-only comparisons, the separate `teacher`
command exports native/source-grid float32 EXR/NPY plus clearly labeled previews.
IPDE's CLI has no dataset or training commands.

`review-dataset` lists samples, splits, available labels, and training readiness.
Use the reported sample IDs with `preview-sample` to write RGB/depth previews;
`--label training` selects the actual distillation target. `curate-dataset`
copies only the IDs supplied through repeated `--keep` options into a new
dataset, preserving their labels and splits without inference.

## Initial two-capture pilot

The supplied IMG_1148 and IMG_1168 originals were processed locally with
DepthPro, V2 Large at 1036, and DA3 GIANT at 1036, using FP32 inference on MPS.
All three produce float arrays rather than 8-bit depth. Source-sized exports
are resampled derivatives, not independent per-pixel measurements.

The workspace includes a two-capture DepthPro dataset, a 160-update real RAFT
fine-tuning run, and a verified export. Its best crop-validation epoch is 2.
Full-grid checks at the same four iterations improved teacher-flow agreement:
IMG_1148 mean error 11.91 to 8.79 pixels; IMG_1168 76.29 to 39.26 pixels.
However, reverse-geometry support decreased on IMG_1168. These two captures are
not independent scene evidence, so the pilot is not a validated replacement
for the original model.

A second dataset uses V2 Large at 1036 with an explicit DepthPro scale anchor.
Its separate 160-update run reduced held-out crop teacher-flow MAE from 90.54
to 30.25 pixels; its best epoch is also 2. Both datasets, runs, and verified
exports appear in the default workspace. This second pilot has the same
two-capture limitation and has not passed independent-scene validation.

The comparison artifacts are under `out/depth-teacher-comparison` and
`out/model-comparison`. Training and full-grid validation evidence is under
`out/raft-studio-pilot`. Model storage precision, inference precision, output
sample counts, and geometric accuracy are distinct concepts.
