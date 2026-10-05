# Dataset library and training collections

Keep a dataset for each photographic subject or experiment. Names and category
labels such as flowers, architecture, or a phone model help organize the library.
They do not mean that every photo in that category is one scene. Optional capture
or scene groups describe related photographs that should stay together when
training and validation are separated. Preserve verified scene groups when
measuring generalization to new scenes.

Select any dataset in the left sidebar and open **Photos and depth**. Selecting a
row chooses the dataset you are viewing. On **Training split**, select one or more
rows and choose **Use for training**, **Use for validation**, or **Do not use**.
The **Use** column displays each dataset's role; highlighting a row and assigning
its role are separate actions. **Dataset actions** contains inspection, storage,
archive, removal and link actions.

Photos are grouped with their teacher targets underneath. Select several photo
rows and use **Remove selected** or **Restore selected**; select a teacher row
to remove just that target. The **Status** column shows **Included**, **Removed**,
or **Some removed** when a photo retains only some teacher targets. Removed
entries remain visible with a strike-through and can be restored after reopening.
**Set split…** moves
the selected entries and their entire known photo/teacher/burst/scene component
to training or validation, including related
entries outside the current filter. Conflicting assignments are rejected.

Membership and split changes save automatically to the selected dataset. These
updates replace only its manifest; existing image and depth arrays remain
byte-for-byte unchanged. Each dataset keeps a durable pending-edit draft while
you switch rows or reopen Studio. A failed save keeps the draft for retry. The
save status distinguishes pending edits, saving, and saved changes; closing with
unfinished edits asks what to do instead of silently discarding them.

**Add photos…** immediately registers the chosen spatial photos in the selected
dataset and opens generation for their depth targets. The registered photo list
survives reopening even when generation has not finished; resume its pending
targets from **Photos and depth**. Existing photos and targets are not regenerated.
**Add from dataset…** adds included photos and teacher targets from another prepared
dataset using its existing array files, without inference.
If imported captures disagree with an existing training/validation assignment,
set a common split for that capture component before saving again.

Train an existing dataset directly from **Train model & compare** when its current
split is suitable. To combine datasets or choose new validation groups, select
datasets in Dataset Studio's **Training split**, choose **Use for training**,
and create a collection. A single available dataset starts with the **Training**
role. Choose **Use for validation** to hold out a whole dataset and switch to
the dedicated validation strategy. If you switch back to an automatic strategy,
change each existing **Validation** role to **Training** or **Not used** before
creating the set. **Do not use** sets its role to **Not used**. Set
**Validation size** to the desired percentage from 0% to 100%; the default is
20%. Percentage and seed changes save automatically to the selected dataset and
restore when you reopen it. The saved automatic setting also applies when you
remove, restore, or add photos. **Set split…** switches to individual assignments
and preserves those choices during later membership edits.
**Apply split and open selected dataset in Trainer** applies that automatic split to the
selected dataset and waits for it to save before opening Trainer. The percentage
applies to independent capture components, so teacher-entry counts can differ.
At 100%, all included groups are validation and no training groups remain;
Trainer reports that training needs a nonempty training split. At 0%, no groups
are held out and Trainer reports that validation examples are required.
**Advanced split options** exposes group preservation and the repeatable seed.
Preparation reuses the original array storage without a full payload copy. Original dataset files
remain unchanged. Every teacher variant of a photo stays on the same side of the
training/validation boundary. Identical source files, identical left RGB planes,
and explicit burst identifiers are protected even when supplied groups are
ignored for an A/B experiment.

Three validation choices are available:

- **Global random:** a seed chooses independent photo/burst/scene components from
  all selected datasets together. A larger dataset will generally contribute
  more held-out examples.
- **Equal per dataset:** each dataset contributes the same count of independent
  validation components. A component can contain multiple photographs or teacher
  variants, so the number of sample entries may differ. Components shared by
  several datasets remain in training. If a dataset has no independently
  selectable photos, use global splitting or add independent examples.
- **Separate validation datasets:** all examples from the selected validation
  datasets remain held out. Any duplicate, burst, or preserved group overlapping
  training causes an actionable error. Remove the overlap before assembling the
  collection.

The seed, selected datasets, original manifest hashes, original splits, grouping
override, and resulting per-dataset counts are recorded in the new manifest.
Category labels never establish independence. Ignoring scene groups is useful
for controlled comparisons but cannot detect unknown related captures.

Dataset Studio can link a dataset directory, or link datasets from another project.
The link stores its existing path in `project.ini`; it does not duplicate large
arrays. Dataset Manager and Trainer show linked datasets in the same library
and training-set selector. Inspection and training read the original location.
Photo membership and split changes update a linked dataset at its original
location as well. Composition and compacting produce new datasets inside the
current workspace. A missing or moved directory
appears as a library warning. Archive applies only to datasets owned by the
current workspace. In **Manage links…**, select links and choose **Remove selected
links** to stage their removal; **Save** applies it. The original dataset files
remain in place. Dataset Studio owns all dataset actions; Trainer's dataset
library is read-only and links back to the manager for review or preparation.

```sh
raft-studio workspace ./workspace --linked-dataset /path/to/existing-dataset
```

## What dataset checks cover

These checks help prevent ordinary mistakes, omissions, corrupted files,
incorrect units, and known training/validation overlap. Hashes identify bytes
and detectable changes; they do not establish who captured an image, whether
labels are truthful, or whether a dataset was intentionally poisoned.
The workflow does not certify an externally obtained dataset as trustworthy.

Before using external material, review its source, license/copyright permission,
consent, intended use, inappropriate content, and possible poisoning separately.
A content classifier cannot grant rights or establish consent, and an unknown
scene relationship can still produce misleading validation scores.

The disabled [external-review hook template](dataset-review-policy.example.json)
sets out a future local review interface. Copy it into your project notes to
record the policy you intend to apply. It is a planning template: Studio does
not execute it, download guard models, or treat its presence as an approval.
Any later integration should keep findings separate from precision-preserved
arrays, record source/model/report hashes and versions, make image uploads
explicit, expose uncertain results to a person, and require human rights/consent
review. Review reports should remain valid only for the source bytes they name.

The command-line equivalents are:

Save an edit description as `edits.json`. For an in-place update, `keep` lists
included sample IDs; omitted IDs remain available as excluded entries for later
restoration. Omitting `keep` preserves current membership. `splits` overrides
selected sample assignments and moves each linked component together. Optional
`validation_fraction` and `seed` apply an automatic split to included components.

```json
{
  "keep": ["sample-1", "sample-3"],
  "splits": {"sample-3": "validation"}
}
```

```sh
raft-studio update-dataset selected-dataset --edits-json edits.json
raft-studio update-dataset selected-dataset --edits-json edits.json --add-dataset more-photos
```

These commands atomically replace the selected manifest. They preserve existing
array files and sample IDs; additions retain their source identity and reuse
existing array storage on the same filesystem. Repeating an addition does not
duplicate its entries. `--expected-manifest-sha256 HASH` rejects changes made
since the dataset was loaded. A running training job uses its captured manifest;
later edits apply to subsequent runs. Conflicting assignments for related captures require one common
split.

For an explicit separate copy, the existing command remains available:

```sh
raft-studio edit-dataset original-dataset --edits-json edits.json --add-dataset more-photos --output-dir edited-version --workers 4
```

Its destination must be new and outside every source dataset. It verifies source
arrays and the completed output before publication and preserves each original
NPY/NPZ/EXR/PNG file's bytes and hashes. Shared storage requires the same filesystem.

For automatic collections:

```sh
raft-studio compose-datasets flowers rooms --output-dir experiment-global --split-mode global-random --seed 42
raft-studio compose-datasets flowers rooms --output-dir experiment-equal --split-mode equal-per-dataset --validation-count-per-dataset 10 --seed 42
raft-studio compose-datasets flowers rooms --output-dir experiment-held-out --split-mode explicit --validation-dataset validation-photos
raft-studio compose-datasets flowers rooms --output-dir experiment-ignore-groups --grouping ignore --seed 42
```

## Lossless storage

New collections reuse each unique original NPY/NPZ/EXR/PNG file through independent
copy-on-write clones on supported macOS filesystems, or immutable hard links on
the same filesystem. No array values or file containers are rewritten. Each
collection has contained array paths and survives removal of its source folder.
Cross-volume preparation requires an explicit `--storage-mode copy`, or direct
training from the existing dataset.

Newly generated datasets and explicit portable copies store each unique float16/32
depth plane in lossless ZIP EXR, and supported unsigned 8/16-bit RGB planes in
lossless PNG. Each float32 teacher result is a single 32-bit channel, with no
RGB duplication or alpha. Masks and other dtypes/layouts use NPZ (a ZIP archive
containing one NPY plane named `data`). Compression changes the file representation; the
array dtype, dimensions, byte order, NaN payloads, signed zero, and numerical
values are checked by exact array hashes. Scientific RGB PNGs preserve their
original sample codes; separately generated preview PNGs remain disposable viewing
images and never become training labels.

New datasets omit native teacher outputs, confidence and unrelated embedded
auxiliary planes by default. Advanced storage switches or CLI
`--retain-intermediates --preserve-auxiliary-assets` retain them explicitly.
Equivalent positive-finite masks are computed from depth; restrictive masks are
still stored. Source photos are never rewritten. Use one teacher without metric
anchoring when relative labels meet the task; additional teachers/anchor maps
retain different numerical data and add to storage. The detail preset now leaves
metric anchoring off by default.

Existing datasets can be copied into this compact form, without removing any
existing scientific planes:

```sh
raft-studio compact-dataset original-dataset --output-dir compact-dataset
```

This verifies all source arrays, preserves split assignments and sample IDs,
deduplicates identical planes, and reports the number of bytes before and after.
It never replaces the original or an existing destination. Already compressed
or incompressible data may not become smaller; the reported ratio describes the
actual result.

## Optional Hugging Face backend

The `datasets` Python library is optional and must already be available in the
configured Python runtime. IPDE does not install it automatically. The backend
uses `datasets.load_dataset` for data-only local or Hub datasets with
`trust_remote_code=False`; loading scripts are unsupported. See the official
[Datasets loading guide](https://huggingface.co/docs/datasets/loading).

An ordinary collection of pictures or relative depth predictions cannot be used
as calibrated RAFT stereo training data. An import must explicitly map:

- Matching native uint8 RGB left and right NPY/NPZ planes.
- A floating-point meter-depth NPY/NPZ plane on the exact left grid.
- Rectified stereo calibration with both camera dimensions, focal length,
  baseline in meters, horizontal principal-point offset, and
  `raft_stereo_ready: true`.
- Whether depth is a user-declared measurement or a teacher pseudo label. Pseudo
  labels must name the genuine teacher checkpoint SHA-256.

The importer accepts encoded NPY/NPZ bytes from Hub columns or relative NPY/NPZ
paths under an explicitly selected local asset directory. It rejects generic
Pillow images, JPEG/PNG display encodings, inferred-dtype Python lists, pickle
arrays, automatic meter-scale conversions, resizing, and row URLs. External
calibration, capture authenticity, and measurement accuracy still require
independent verification; the importer does not claim an external photo came
from an Apple camera.

A minimal mapping file is:

```json
{
  "columns": {
    "left": "left_npy",
    "right": "right_npy",
    "depth": "depth_npy",
    "calibration": "camera",
    "category": "subject",
    "metadata": "photo_metadata"
  },
  "depth_units": "meters",
  "depth_role": "measured",
  "coordinate_reference": "spatial_left"
}
```

Optional column keys are `id`, `source_sha256`, `scene`, `category`, `metadata`,
and `burst_ids`. Mark `verified_scene_groups: true` only when the mapped `scene`
column describes verified independent scenes. `category` is an organizational
annotation and does not affect split grouping.

For pseudo labels, replace `depth_role` with `pseudo_label` and add
`teacher_metadata` containing `checkpoint_sha256` and the model identifier.
For measured references, use the trainer's supervised mode. Importing a dataset
does not turn measured labels into a teacher checkpoint.

```sh
raft-studio import-hf owner/calibrated-stereo --mapping-json mapping.json --split train --revision COMMIT --output-dir imported-stereo
raft-studio import-hf json --data-files rows.jsonl --asset-dir ./source-arrays --mapping-json mapping.json --output-dir imported-local
```

A local JSON row can hold `left_npy: "capture-1-left.npy"`,
`right_npy: "capture-1-right.npy"`, and `depth_npy: "capture-1-depth.npy"`, together
with the explicitly mapped calibration object. The output must be outside the
asset directory. Malformed rows are skipped and listed by row index and reason;
if no compatible rows remain, no dataset is published.
