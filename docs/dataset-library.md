# Dataset library and training collections

Keep a dataset for each photographic subject or experiment. Names and category
labels such as flowers, architecture, or a phone model help organize the library.
They do not mean that every photo in that category is one scene. Optional capture
or scene groups describe related photographs that should stay together when
training and validation are separated. Preserve verified scene groups when
measuring generalization to new scenes.

After reviewing individual photos and teacher variants, select several datasets
in the Training Set tab and create a new training collection. Original datasets
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

The command-line equivalents are:

```sh
raft-studio compose-datasets flowers rooms --output-dir experiment-global --split-mode global-random --seed 42
raft-studio compose-datasets flowers rooms --output-dir experiment-equal --split-mode equal-per-dataset --validation-count-per-dataset 10 --seed 42
raft-studio compose-datasets flowers rooms --output-dir experiment-held-out --split-mode explicit --validation-dataset validation-photos
raft-studio compose-datasets flowers rooms --output-dir experiment-ignore-groups --grouping ignore --seed 42
```

## Lossless storage

New collections store each unique plane once using NPZ (a ZIP archive containing
one NPY plane named `data`). Compression changes the file representation; the
array dtype, dimensions, byte order, NaN payloads, signed zero, and numerical
values are checked by exact array hashes. Preview PNGs are disposable viewing
images and never become training labels.

Existing datasets can be copied into this compact form:

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
