# Material import, crop grid and training throughput

Build: Texture Studio 0.9.24 (119), 2026-10-12.

## Resulting behavior

- AO, ambient occlusion and packed supplemental maps cannot supply diffuse input. Material import and diffuse-classifier discovery use the same explicit color-role parser. Named AO files also cannot hide standalone `BaseColor.png` and `Height.png`. Supplemental maps produce no broken-pair warning.
- The selected dataset grid contains only exact crops at that size. Undersized original references remain registered and become visible when an appropriate smaller grid is selected. Prepared datasets report their actual crop grid and validate color variants as well as target maps.
- Maps in the same explicitly named tier may differ by up to eight edge pixels. Preparation uses their common native intersection, physically crops each map where necessary, and preserves original bytes. It never scales, interpolates or pads an undersized source.
- Folder previews read PNG headers and file statistics, calculate only the selected crop plan, and skip downloads and unrelated inventory maintenance. Import verifies complete PNG chunks, hashes and provenance before registration. Changed or corrupt maps cannot be registered using a fast preview. Preview and import do not invalidate their own proof through automatic inventory maintenance.
- Recognized groups rejected for naming conflicts report their actual conflict and source paths, avoiding redundant per-map pairing errors.

## Real dataset evidence

The `/opt/ipde/sources_mats` preview found 456 sets in 0.6177–0.6568 seconds across three runs. Its three remaining warnings identify duplicate `pine_bark` names; there are no AO pairing warnings. Repeating the scan against the populated BadMonkey dataset took 0.7775–0.7885 seconds and identified all 456 sets as already registered without changing its index.

A read-only BadMonkey audit found 456 registered originals, including 143 smaller than 2048. The 2048 view contains 313 exact 2048 crops and does not contain `anti_skid_tiles_1k_full`. Dataset registration and review metadata remained unchanged. Original source preservation and exact per-channel crop bytes are also covered by regression fixtures.

Evidence: [empty-dataset scans](material-quick-scan-2026-10-12.json), [populated-dataset scans](material-existing-dataset-quick-scan-2026-10-12.json), [crop audit](material-badmonkey-crop-proof-2026-10-12.json).

## Training evidence

Compact training retains compiled derivative programs, reuses parameter feeds only while their complete tensor values are unchanged, and omits unused prediction readback during gradient steps. Empty or malformed parameter sets still fail validation. Validation and prediction use separate cached programs without evicting the gradient program. CPU Float32 decoding uses a bulk copy with exact bit preservation.

The isolated CPU accumulation/AdamW benchmark reduced median overhead by 42.1% at accumulation 1 and 46.6% at accumulation 2. All before/after final parameter and optimizer-moment hashes match. This measures CPU overhead, not total training speed.

The procedural 512-grid, width-32 GPU benchmark observed 162.63 ms/sample at accumulation 1 and 148.42 ms/sample at accumulation 2, approximately 9.6% more samples per second. Each run compiled once; accumulation 2 reused parameter feeds within each update. This supports amortizing feed and optimizer overhead, rather than a train/evaluation mode switch. Prediction readback removal alone changed total time by less than 1%, within single-run noise.

These throughput measurements exclude PNG loading, crop preparation, DepthPro, production 2048 training and learned-quality qualification. Evidence: [CPU comparison](native-training-cpu-overhead-2026-10-12.json), [compact GPU measurements](native-compact-throughput-2026-10-12.json).

## Regression checks

The final focused Release checks exercised 121 cases across dataset lifecycle, compact model, crop sizing, native dataset, filename roles, import diagnostics, optimizer and quick scan suites. There were two opt-in skips. The lifecycle fixture's obsolete expectation of a missing display transform was replaced with assertions for its identity crop, exact integer-code algorithm and original source hash; that case then passed on a focused rerun. The other 118 cases passed in the preceding run, giving 119 passing cases across those checks.

The broader Release run also passed the crop-recovery, import-readiness, source-recovery, native trainer and compact trainer suites. Trainer coverage includes gradient accumulation, partial accumulation on stop, validation, checkpoint continuation, deadlines, cancellation and preserved completed outputs. The opt-in production training benchmark was not run. Native build-tool checks and `git diff --check` passed.

App-hosted Release tests use testability and disable hardened runtime for XCTest injection into the ad-hoc test host. Production packaging uses the normal optimized Release settings and hardened runtime, with the generated test plug-in moved out of the app before packaging.

## Installed package

`make package` succeeded and created `dist/Texture-Studio-macos-arm64.zip`. Native bundle validation, matching version checks and deep strict signatures passed. The production parent has `flags=0x10002(adhoc,runtime)`, and no XCTest plug-in remains in the app.

The closed-app installer installed `/Applications/Texture Studio.app` at version 0.9.24, build 119. A recursive comparison found no differences from the packaged build; the installed deep strict signature check passed. Running the installed executable with `--smoke-test` exited successfully and printed `Texture Studio native smoke passed`.

The installed Material Dataset app was launched with BadMonkey. Native visual automation was unavailable (`Sky Computer Use native pipe startup failed`), so no visual click-through is claimed. The crop projection is verified by compiled tests and the read-only live dataset audit above.
