# RAFT Studio

RAFT Studio is the separate Qt app for creating and managing spatial-photo
datasets, distilling a depth teacher into RAFT-Stereo, and exporting checkpoints.
IPDE remains the extraction app: choose an exported checkpoint in its existing
**RAFT model** field and continue extracting images and depth maps.

Launch the installed local build:

```sh
make studio
# Or open build/RAFT Studio.app
```

The default workspace is `/opt/ipde/raft-workspace`. Its `datasets`, `runs`, and
`exports` directories contain ordinary local files. Dataset creation and training
run in a background Python process; progress appears in the app and a task can
be cancelled. Model inference never uploads photographs or downloads weights.
See [model setup](model-setup.md) for explicit, revision-pinned downloads.

## Create a dataset

Add spatial HEICs in the **Create dataset** tab. Edit the group column so related
captures of one scene have the same group. Keep left/right views, repeat shots,
and variants of a photograph together. Metadata detects exact duplicates and
reported bursts; it cannot reliably determine whether two photographs depict
the same room. Capture-group validation is not independent scene validation.

Choose the teacher and its local checkpoint/source directories:

- **DepthPro** provides estimated meter depth and is the initial RAFT teacher.
- **Depth Anything V2 Large** provides relative inverse depth. Input size 1036
  offers a larger processing grid than its default 518.
- **Depth Anything 3 GIANT 1.1** provides relative depth in the current adapter.
  The app does not claim that a larger model guarantees better geometry.

Relative predictions require an explicit metric anchor before becoming physical
RAFT disparity labels. Anchoring to another model still produces pseudo-labels;
it does not establish measured accuracy. Raw relative predictions remain stored
separately from any anchored derivative.

For a detailed initial teacher, choose **Depth Anything V2 Large** with input
size **1036** and **Anchor relative depth with DepthPro** enabled. The anchor
fits estimated scale on the same RGB grid and rejects poorly agreeing fits.
Compare its results with DepthPro and DA3 on your own scenes before settling
on a teacher. DA3 also supports paired/multiple-view inference upstream; the
current Studio adapter evaluates single views independently.

Datasets preserve all decoded RGB/auxiliary arrays as exact NPY files, camera
calibration, source/model hashes, source revisions, native prediction grids,
source-grid float32 predictions, masks, and grouping/split metadata. Depth
targets never pass through an 8-bit preview. Normalized RGB tensors used by a
model are separate from the saved raw data. Training checks array hashes before
loading targets.

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
source folder, and use **Train RAFT**. Update-block training is the initial
setting; full-model fine-tuning is optional. Native crops preserve pixel units
instead of downsizing disparity labels. Each epoch is evaluated on held-out
groups, and the best checkpoint is retained, including the original baseline
if every candidate becomes worse.

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
