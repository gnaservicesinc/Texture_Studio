# Dataset migration and duplicate removal

Verified on 2026-10-04 with full file and decoded-array checksum checks. No
workspace dataset was changed or deleted.

Keep the original dataset:

`/Users/andrewsmith/Documents/ipde/iPhone16 pro max spacial/workspace/datasets/dataset-20261004-034119`

The existing duplicate can be removed when no running training/review operation
is using it:

`/Users/andrewsmith/Documents/ipde/iPhone16 pro max spacial/workspace/datasets/iphone-da3-20261004-045241`

Both contain the same 2,403 unique array files, totaling 21,507,801,203 bytes.
All file hashes, decoded array hashes, dtypes and shapes match. The duplicate
contains no unique image, depth target, auxiliary array or calibration. All 171
samples have equivalent target and raw-data metadata. Do not remove the whole
project folder: it also contains the original dataset.

The duplicate's unique information is its customized split and composition
provenance: 10% validation, random seed 84785, and ignored authored groups. The
original uses 20% validation; 45 sample entries have different split assignments.
The exact duplicate manifest has been saved losslessly in
[iphone-da3-20261004-045241.dataset.json.gz](/opt/ipde/ipde/out/training-readonly-20261004/migration/iphone-da3-20261004-045241.dataset.json.gz).
Decompression was verified byte-for-byte. This 3,479,479-byte backup records the
old split; it is not a standalone dataset without the retained arrays.

For migration:

1. Finish or stop any run using the old location.
2. Copy the project folder, retaining `project.ini`, the original dataset,
   `workspace/runs`, `workspace/exports`, and any project export folders. Exclude
   the duplicate array folder above; include this manifest backup if keeping its
   experiment history. Keep the original HEIC photos separately for future
   regeneration and comparisons.
3. Open the copied project in IPDE Studio. Select the retained dataset and use
   **Inspect selected** to verify its arrays before removing the old location.
   Relink any external datasets and select available RAFT source/model paths on
   the destination machine if their locations changed.
4. Train the existing dataset directly using **Start model training**. No
   preparation copy is needed. Metadata screening finds 165 usable targets
   (131 training, 34 validation); six relative targets are skipped automatically.
   Geometry/crop screening can exclude further unusable targets. Dataset files
   remain unchanged.

Future **Prepare training set** operations reuse arrays through copy-on-write
clones or immutable hard links on the same filesystem. Prepared folders contain
their own array paths and survive removal of their source folder. When moving
across volumes, copying one complete dataset is sufficient; storage sharing may
not carry across the volume boundary. Existing older duplicate folders are not
automatically reclaimed.

At verification time, the workspace's `runs` and `exports` folders were empty.
No model training was launched during these checks.

Verified manifest SHA-256 values:

- Original: `853b557e29876403594c9f16c8dc502463cea22f2b2c9660017437bde7306acb`
- Duplicate: `0211bf886c6a3054720055c37a280f975eb29e86fd10b0d257fff3ca187fb180`
