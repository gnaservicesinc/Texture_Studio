# Dataset workflow repair validation — 2026-10-09

Installed `/Applications/Texture Studio.app`, version 0.9.12, build 103, Release, with all four bundled material tools. The installed signatures and source files match the final built bundle. Changes remain local in the repository.

## User workflow

1. Open Material Dataset and the existing `/opt/ipde/sources_mats/Matts` dataset.
2. Open Dataset Info and choose Rendering & training resolution (256, 512, 1024 or 2048).
3. Choose Import Folder and select `/opt/ipde/sources_mats`. Review the crop/split counts and source notices, then press Import.

Raw folders can also be opened or dropped directly. A folder without a dataset opens setup if no dataset is selected, or an import preview for the selected dataset. The first full source verification took about 70 seconds for this 33 GB collection; progress identifies the folder. Changing resolution after scan does not reread all images. A duplicate scan took 0.43 seconds.

## Real collection evidence

The real source collection was imported into an isolated temporary dataset. All 231 source sets across 116 folders were recognized. Their registered SHA-256 hashes matched the independent initial full-source scan. The original user's Matts dataset was not imported or reconfigured by the test.

At 2048 square: 230 source sets supply 240 native crops. One smaller source remains inspectable but cannot train at this size. Roughness and normals each have 227 training / 13 validation crops. Displacement has 221 training / 13 validation crops and 6 crops without a supported displacement target. Preview and imported plans match. Reimport detects all 231 duplicates.

Import, dataset rendering and preparation use one metadata-only crop/split planner. Prepared synthetic numeric crops were compared element-for-element with their source regions. Original source maps are referenced unchanged; import applies no resizing, padding, gamma or normalization.

## Checks

- Full Python suite: 920 passed, 224 subtests passed, 17 existing training-hook warnings.
- Native suite: 167 executed, 1 optional RAW fixture test skipped, 0 failures.
- After the final isolated-entrypoint fix, all 6 backend packaging tests passed, including direct `python -I -B` create/scan/import/reopen and model capabilities.
- Release build and all bundled tools signed and staged successfully.
- Installed source parity, direct isolated dataset/model commands, and app smoke passed.
- Native-host rendering snapshots were inspected for the new setup, info and import sheets. Live computer control was unavailable; OS screen capture also lacked Screen Recording permission. A manual click-through is therefore not claimed. Cached native snapshots have incomplete glass-button compositing.

Logs are stored beside this report. These checks establish the import/setup contracts, not trained model quality or extended training stability.
