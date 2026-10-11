# On-demand crop refresh — 2026-10-11

The reported stall occurred before any optimizer update. The local aborted run had 365 training pairs at 1024×1024, with DepthPro Height Refiner, every-epoch flips and center spins enabled. During about 70 seconds of “Refreshing cached crops,” it generated 19 DepthPro inputs and remapped 13 of them. The status stayed at 0/365 while the trainer refreshed the entire training inventory. PNG encoding already used zlib level 1.

Training now requests one paired crop inside the active update. Preparation is memoized once per run or epoch, according to the selected timing. Shared color variants retain matching orientation and a shared target. A deadline or Stop & Save finishes the active map and its available accumulated gradients, then validates and saves; it schedules no further crop preparation. Baseline and final validation still use the full configured validation pool.

Unchanged crops preserve their exact PNG bytes and prediction identity. Flip-only work no longer encodes an unnecessary unflipped copy. Center spins can still need that intermediate to predict the fresh field before an optional flip. Intermediate files are removed on cancellation. Progress identifies crop transforms, DepthPro input preparation and writes while retaining the active update.

Crop history appends one JSONL record per refreshed pair. The complete JSON history is written at checkpoint and return boundaries, avoiding a growing whole-history rewrite on every map.

One DepthPro predictor remains available during the run and releases on return. This allows requested fields to be prepared without processing the entire dataset first, at the cost of overlapping Core ML and refiner residency. Model acquisition remains setup work; requested crop preparation counts toward the training deadline. These changes do not qualify trained material quality or 2K model training.

An optimized standalone comparison used the original and current augmentation code with two RGB16 2048×2048 color variants and one shared gray16 height target, producing three PNGs. Means of three runs:

| Refresh | Original | Current |
| --- | ---: | ---: |
| Unchanged, first epoch | 2.441 s | 0.068 s |
| Unchanged, next epoch | 2.431 s | 0.048 s |
| Both-axis flip, first epoch | 2.514 s | 1.552 s |
| Both-axis flip, next epoch | 2.497 s | 1.542 s |

Integer pixels matched the expected transformations, progressive flips restored the original pixels, source hashes stayed unchanged, and current unchanged copies retained exact compressed bytes and SHA-256. DepthPro was disabled in this comparison; these timings describe PNG and file work only.

The real upstream DepthPro model completed a 256×256 one-step training/export/inference check in 13.95 seconds on Apple M2 Max. It generated one prediction, reused the cached field, and returned bitwise-identical normalized input at inference. Original diffuse and height hashes stayed unchanged. Process peak resident memory was 8,071,086,080 bytes. This verifies actual Core ML prediction and MPSGraph optimization with the on-demand predictor lifetime at that grid; it is not a larger-grid training or material-quality qualification.

The focused native batch completed **64 tests, three optional larger-grid benchmark skips, zero failures** in 76.903 seconds. It covers stopping after one requested pair, leaving unscheduled pairs untouched, per-run/per-epoch reuse, shared variant alignment, progressive flips, center spins, cache provenance, checkpoint/export behavior and progress counters. Tests used optimized Release code with `ENABLE_TESTABILITY=YES` and `ENABLE_HARDENED_RUNTIME=NO` for the local test host. Delivery uses the normal application build settings.

The normal Release build succeeded and Material Trainer launch verification passed. Native bundle checks and parent/nested code signatures passed. The app retained hardened runtime, contained no test bundle, and installed at `/Applications/Texture Studio.app`. A recursive contents comparison with the build output returned no differences. Both main executables had SHA-256 `3001cb6aac47d6623486c4f57cf749de108c603b2ac4a62e4f1800b495e6e82a`.

The installed executable returned **`Texture Studio native smoke passed`** with exit code zero. Local test, benchmark, real DepthPro, build, install and smoke evidence is retained under the Texture Studio application-support folder at `Validation/crop-refresh-2026-10-11`.
