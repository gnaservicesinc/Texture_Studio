# IPDE Studio

Launch `make studio` (or `build/IPDE Studio.app`). Open/create a project, then
choose **Effect / map**, **Depth estimation**, **Photo effects / masking**, or
**Manual**. Guided modes hide technical controls; **Advanced settings** exposes
individual overrides. Manual exposes all choices. Each project stores its
workspace and `project.ini`; datasets and models are not silently selected into
another project's extraction settings.

Studio launches **Extract maps**, **Manage & review datasets**, and **Assemble &
train**. The manager and trainer have separate windows focused on their step.
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

**Scan spatial directory** recurses without following links. Nonspatial photos,
portraits without calibrated stereo, malformed containers and inconsistent views
are skipped with a reason. Apple camera metadata and supported stereo structure
are required. These checks cannot establish authenticity of convincingly forged
metadata or prove an image was never AI edited.
Choose the folder containing original HEIC/HEIF/HIF photos, rather than the
project's workspace or generated dataset folder. Validated photos appear in the
list as the scan runs; cancelling keeps the photos already found. Scanning and
manual addition share the same duplicate handling.

Select one to three teachers. Each teacher creates an independently reviewable
entry for a photo, sharing its raw data and validation split. Uncheck a poor
teacher result without dropping the photo's other results. Generation publishes
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

The optional **Prepare training set** step accepts many datasets. Choose a dedicated
validation dataset, seeded random selection across all datasets, or equal random
counts of independent groups per dataset. Group preservation/override enables
A/B experiments; exact duplicates, teacher variants and known bursts always stay
together. External validation overlapping training sources is rejected. Use
**Compare selected model with baseline** on a spatial photo to inspect the
original and trained RAFT results with the same grid and meter contrast.
With a single available dataset, **Use for training** starts checked. The
message above **Create training set & continue** explains missing selections,
name conflicts, unsaved review exclusions, or an active task. Save a reviewed
copy after excluding samples, then select that copy for the training set.
Creating a set verifies the selected arrays and reuses their existing storage.
On supported macOS filesystems it creates independent copy-on-write clones;
otherwise it uses immutable hard links on the same filesystem. It does not
recompress or make another full array copy. Verification can still take several
minutes for large datasets.
The button reads **Creating training set…** during this work; the progress
message reports reused arrays and final verification. Completion selects the saved
set and opens the model training step. Preparing a set assigns validation splits;
**Start model training** runs the optimizer and writes a checkpoint.

Studio windows fit within the screen's usable area. Scroll to reach controls
on smaller displays; review captions have their own space beneath the images.

## Storage and external datasets

New GUI datasets use lossless NPZ (ZIP deflate of an NPY plane) with exact
round-trip checks. Identical arrays are stored once even when several records or
teacher variants use them. Raw arrays are never normalized or gamma corrected.
Prepared training sets retain the original NPY/NPZ files byte-for-byte through
shared storage. Their array paths stay inside the new set, so removing the source
directory does not break the set. Hard-linked payloads must remain immutable;
Studio's array writers publish new files rather than editing existing ones.
The storage column reports reused arrays and newly added metadata. Sources and
prepared sets must be on the same filesystem; train a linked dataset directly
when its files are on another volume. The CLI's explicit
`compose-datasets --storage-mode copy` creates a portable compressed copy when
requested. Existing older copies are left in place.
**Compact dataset** creates a new verified compressed copy of an older dataset;
it preserves split assignments and leaves the original available. Using both
copies consumes additional storage until you remove the original yourself.

**Include display-camera teacher** is an advanced opt-in. It retains extra full
resolution predictions/anchors that may be useful for registration diagnostics
but can add hundreds of MiB per photo. The original display RGB remains lossless.
Training uses a bounded cache of decoded samples instead of loading the entire dataset into RAM.
NPZ reads decompress a requested plane into memory; NPY remains available with
`dataset --uncompressed` when memory mapping is preferable.

Optional Hugging Face imports use the `datasets` extra with explicit column
mapping, NPY/NPZ planes, meter depth and compatible stereo calibration. Generic
image or text datasets cannot directly supervise calibrated RAFT. See
[dataset library](dataset-library.md) for mappings and examples; the backend uses
[data-only dataset loading](https://github.com/huggingface/datasets/blob/main/src/datasets/load.py).

## Model and precision background

RAFT Studio is the separate Qt app for creating and managing spatial-photo
datasets, distilling a depth teacher into RAFT-Stereo, and exporting checkpoints.
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

Add spatial HEICs in the **Create dataset** tab. A **teacher** is the depth model
that generates the target maps RAFT will learn to imitate. Its targets are
estimates: a detailed-looking map can still contain incorrect surfaces or
distances. Review the maps before training.

Edit the group column so related captures of one scene have the same group.
For example, several photos of the same room or object during one capture
session belong together, even if the camera moves. Keep repeat shots and
variants together; each HEIC's left/right views already stay together. Use
different group labels for scenes that really are independent.

**My groups separate independent scenes** records your
confirmation that you have grouped the photos this way. It lets the report
describe the held-out groups as scenes. The app does not automatically verify
scene identity, and checking the box does not verify depth-map quality. Do not
check it just because photos have different filenames or group labels.

Grouping prevents **validation leakage**: if RAFT trains on one photo of a room
and is tested on another very similar photo, its score may look good without
showing that it works on a new scene. Metadata catches exact source/RGB
duplicates and reported bursts, but cannot reliably recognize the same room or
object. Leave the box unchecked when scene independence is uncertain; the
report then uses the default capture grouping. Supplied group labels still
keep related captures in the same split when the box is unchecked.

Choose the teacher and its local checkpoint/source directories:

- **DepthPro** provides estimated meter depth and is the initial RAFT teacher.
- **Depth Anything V2 Large** provides relative inverse depth. Input size 1036
  offers a larger processing grid than its default 518.
- **Depth Anything 3 GIANT 1.1** provides relative depth in the current adapter.
  The app does not claim that a larger model guarantees better geometry.

**Depth scale — Estimate meters with a second model (DepthPro)** converts a
relative teacher's values into an
estimated meter scale using DepthPro. V2 and the current DA3 adapter describe
near/far relationships, but their values have an image-specific scale: a value
of 2 does not mean 2 meters or necessarily the same distance in two photos.
The anchor runs DepthPro on the same RGB image and fits a scale and offset in
inverse depth. It creates a separate estimated-meter map from the relative
prediction; the original teacher values remain unchanged.

Use the anchor when you intend to train RAFT from **V2 or DA3** in this workflow:
RAFT's physical stereo labels require meter-depth estimates and camera
calibration. The option is disabled for **DepthPro**, which already estimates
meters. You can leave it off when creating relative maps for visual review or
teacher comparisons, or when you deliberately want to retain only the relative
prediction. An unanchored V2/DA3 dataset cannot train RAFT through this
distillation workflow. Independently measured meter-depth references are a
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

**Input size** controls the teacher's processing resolution, not the stored raw
photo quality. V2 uses it for the shortest side; DA3 uses it for the longest
side. 1036 gives V2 a larger processing grid than 518, at a cost in time and
memory, but does not guarantee better geometry. DepthPro uses a fixed native
1536×1536 prediction grid, so this control is disabled for it. Compare results
on your scenes before settling on a teacher and size. DA3's current Studio
adapter evaluates single views independently.

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

## Review depth maps and exclude poor samples

**Generate & review dataset** opens the review tab after generation. To
inspect an existing dataset, select it in the library and choose **Review depth
maps**. Select a sample to see its RGB image beside its generated depth map.
Click either image to open a native-pixel preview centered on that point. Drag
to pan, and right-click or click outside to dismiss it. Hover for a synchronized
1:1 magnifier.
Look for incorrect object boundaries, flattened or invented surfaces, holes,
and inconsistent near/far order. Clear the sample's **Include** checkbox to
exclude it from the reviewed copy. This removes the whole training sample,
including both stereo views, rather than only its preview image.

Start with **Selected training target**, which shows the map this sample would
actually use for distillation. Depending on the sample, that can be the direct left-view
teacher, the accepted anchored teacher, or a registered display-image teacher.
The label selector also exposes the raw teacher and available anchor/display
maps so you can inspect the source of a problem. Viewing another label does
not change which target training uses. A relative or rejected-anchor target
remains unsuitable for physical RAFT labels even if its preview looks good.

The grayscale depth preview uses **near = white, far = black**. **Magenta** marks
invalid or unsupported values. Each preview uses its own display range, so
matching shades in different photos do not imply matching distances. The
preview is a visualization only: the original float arrays, masks, and raw
RGB/auxiliary data are untouched. A visually plausible map still needs
independent accuracy checks for displacement work.

Choose **Save reviewed copy** to create a new dataset containing only included
samples. It copies the existing data without running inference again and
preserves the original dataset, sample IDs, groups, and train/validation splits.
Train from the reviewed copy in the library. Studio asks you to save unchecked
samples before training the dataset you are reviewing, so unsaved exclusions
cannot be silently ignored. Keep usable samples in both
splits. If all validation samples are excluded, create another dataset with
suitable held-out groups before training; the review does not silently move
training samples into validation.

## Display image and stereo alignment

The HEIC display image can come from either stereo camera and has different
resolution, framing, and processing. RAFT's native correspondence reference is
the decoded left view. Scaling all three RGB images to the same size and
overlaying them is a useful alignment diagnostic; it is not a depth registration.

RAFT Studio generates an independent display teacher and retains its grid. A
same-camera feature registration uses held-out matches and spatial residual
checks. Only validated regions can supply display-derived left-grid labels.
Unsupported regions remain NaN. The direct left-view teacher is retained and is
used when the display registration is rejected; no display depth is stretched
into the left grid. A registered right-camera target needs additional stereo
reprojection and is not silently treated as left-camera depth.

For a positive metric left-grid target, training uses:

```text
geometric_disparity = focal_pixels * baseline_meters / teacher_depth_meters
signed_flow = (cx_right - cx_left) - geometric_disparity
```

The presentation-only HEIC disparity adjustment does not enter this formula.
Out-of-image/crop correspondences and estimated occlusions are excluded.
Optional photometric support gives a stricter training mask. A mask does not
turn an estimated label into independently measured depth.

## Train and export

Select a dataset from the library, choose the original RAFT checkpoint and
source folder, and use **Start model training**. Update-block training is the initial
setting; full-model fine-tuning is optional. Native crops preserve pixel units
instead of downsizing disparity labels. Each epoch is evaluated on held-out
groups, and the best checkpoint is retained, including the original baseline
if every candidate becomes worse.

Training reads dataset files without modifying them. Targets without an accepted
meter scale or usable stereo correspondence support are automatically excluded
from the run. The run report records their sample IDs and reasons, plus the
actual training and validation samples. Usable samples retain their original
splits. Dataset integrity and split leakage checks still apply, and training
needs usable samples in both splits. Checkpoints and reports are written outside
the input dataset, normally in the workspace's `runs` folder.

The status shows dataset checking, target preparation, model setup, baseline
validation, the current epoch and step, held-out validation crops, and checkpoint
writing. The update counter advances after each optimizer step; setup and
validation do not count as model updates. Completion identifies the saved
checkpoint and best epoch.

The training controls have these meanings:

- **Train scope — Update block:** adjust only RAFT's iterative correction
  component. This is the initial choice for a small experiment. **Full network**
  adjusts all model weights, giving more freedom but requiring more resources
  and greater care to avoid overfitting a small dataset.
- **Epochs:** how many rounds of updates and held-out evaluation to run.
- **Steps / epoch:** how many optimizer updates occur in each round. Each step
  draws one random training image and a crop; an epoch is not necessarily a
  pass through every image. Total updates are epochs × steps per epoch.
- **Patch pixels:** the requested side length of a square crop at the original
  stereo resolution. Larger crops provide more context and cost more memory. Use a
  multiple of 32, at least 64; the labels are cropped rather than downsized.
- **RAFT iterations:** how many successive correspondence refinements RAFT
  makes for each crop during training and evaluation. More refinements cost
  time and memory and are separate from optimizer steps or epochs.
- **Device:** where training runs, with the same hardware meanings as the
  inference-device control. Training memory needs can exceed inference needs.

The validation score is agreement with teacher-derived flow on fixed crops.
It is not an absolute accuracy measurement. Checkpoints remain experimental;
evaluate full images and independent scenes before treating one as your device
model. The same checkpoint can then be reused in IPDE. Training again is only
needed when testing exposes a gap, the capture pipeline changes, or new useful
data becomes available.

**Export selected model** creates a new directory with `raft-model.pth` and
`model.json`. Export verifies architecture loading and bit-exact weight
preservation. Architecture metadata makes loading independent of the filename.
The export excludes source photographs and never changes IPDE's active model.

## Separate command-line interface

```sh
.venv/bin/python raft_studio.py --json workspace /opt/ipde/raft-workspace
.venv/bin/python raft_studio.py dataset photo-a.HEIC photo-b.HEIC \
  --output-dir /opt/ipde/raft-workspace/datasets/my-device \
  --model depthpro --groups scene-groups.json
.venv/bin/python raft_studio.py dataset photo-a.HEIC photo-b.HEIC \
  --output-dir /opt/ipde/raft-workspace/datasets/my-device-v2 \
  --model depth-anything-v2 --input-size 1036 --metric-anchor depthpro \
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
  --epochs 10 --steps 16 --patch-size 256 --device mps
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
