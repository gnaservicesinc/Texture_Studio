# Preparation performance and training crash validation — 2026-10-09

Texture Studio 0.9.12 build 105 moves the application, tests and Xcode project to `src/TextureStudio`. Build, run and version-check tools use the new path. The implementation and bundle remain independent of Python; no interpreter environment or model download was created for this work.

## Crash and memory fixes

The 08:49 crash report ended in MPSGraph automatic differentiation. A native LLDB reproduction isolated `MPSGraphTileOp` in decoder nearest-neighbor upsampling. Repeating samples through concatenation preserves their Float32 bits and provides the expected derivative: four repeated samples contribute four copies of the incoming gradient.

Inference and validation use one forward graph and construct no optimizer state. Actual updates create a separate graph lazily, with frozen generator/encoder features retained on the GPU between stages. These boundaries disconnect only paths without trainable adapters; all paths influenced by the selected decoder still contribute gradients. Computation remains Float32 with reduced-precision fast math disabled.

Compiled results are mapped using their tensor identities, following Apple's [MPSGraphExecutable output ordering contract](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutable/targettensors). The model's program cache no longer forms a reference cycle: programs share immutable weight snapshots without retaining their owning model. A regression verifies that both objects deallocate.

## Preparation performance and precision

Independent materials run concurrently, bounded by CPU count and a conservative source/crop memory budget. A worker decodes each original map once for all its crops. Cancellation joins every writer before releasing locks or deleting staging files; publication remains atomic. Activity and worklogs report material counts and worker count.

PNG processing uses scoped byte buffers, row copies and in-place filter restoration. Lossless Sub filtering and faster compression preserve every integer code. Exports are decoded and compared to the original crop before publication. Distinct color variants with identical file bytes retain separate filenames and training membership; the canonical input is selected by source path and checksum together.

| Release measurement on existing maps | Result |
| --- | --- |
| Four 4096 × 4096 material families, twelve original maps → twelve 2048 × 2048 crop files | 10.096 s with one worker; 3.486 s with four workers: 2.90× faster |
| Original checksums and generated files across worker counts | All twelve originals unchanged; all twelve generated PNG hashes identical |
| Complete preparation benchmark resources | 3.29 GB maximum RSS; zero swaps |
| Paired three-run PNG decode/crop/encode/tensor measurements on 4K height, normal and diffuse | Combined phases 1.18–1.85× faster; green-channel conversion about 16× faster |
| PNG precision checks | Original/crop integer bytes, Float32 model tensors and decoded exports match exactly |
| Pinned tensor archive reconstruction | All 2,896 tensor names, types, shapes and bytes match the previous reader; combined inventory SHA-256 `2ae9c6fc150330e93b207a73d9461319fb8a2c905bedfafd52e5a990a99c1db2` |

These are local measurements, not throughput guarantees for every dataset or map format. The preparation comparison uses the final lossless PNG code for both worker counts. Faster compression alone is not uniformly faster on every map: the tested RGB16 diffuse encoder took slightly longer while producing a smaller file.

## Model and application validation

- Ten focused model tests cover both training scopes at rank 64, full operation sequences, exact upsampling bits/gradients, validation without autodiff, result identities and cache deallocation.
- The installed pinned learned base (`3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d`) completed real map-decoder updates at 64 × 64 and 256 × 256 with rank 64/alpha 16. All 578 factor outputs were returned, including a changed factor before decoder upsampling. Nonzero-adapter forward outputs match the original graph byte-for-byte on both grids.
- The learned-base replay's maximum RSS was 4.85 GB; macOS reported a 21.36 GB peak memory footprint across both grids and compilation. These are distinct measurements. Production 2048 × 2048 training, long-run throughput and material quality were not evaluated; the user's selected grid remains unchanged.
- Sixteen focused dataset tests cover bounded concurrency, deterministic crops, duplicate-byte variants, cancellation, failure cleanup and progress events. Five PNG tests include 240 independently constructed filter/Adam7/precision/channel fixtures.
- Full Xcode suite: **264 passed, one optional original-RAW-photo fixture skipped, zero failures**. Result: `build/NativeMigration/Logs/Test/Test-TextureStudio-2026.10.09_09-22-55--0400.xcresult`.
- Native build-tool, version and transactional installer regressions passed.
- Optimized arm64 Release built and installed at `/Applications/Texture Studio.app`, including all four material tools. Installed contents match the built suite byte-for-byte, its strict code-signature and interpreter-free bundle checks passed, and its native smoke test passed. The suite is about 58 MB. Package SHA-256: `e110ec8df316c7b516d452e32896737e1a5e135b15526579c2661598d2a7d212`.

Detailed local logs and benchmark records are under `out/native-speed-validation/`. Original photographs, datasets, map precision and model weights were preserved.

## Follow-up dataset fixes and streaming crop measurement

The earlier build-105 timings above measure the previous full-source decoder and multi-target preparation. The follow-up implementation stages only diffuse variants and the selected output target, omits base64 profile payloads and duplicated source inventories from compact metadata, and uses scanline streaming for source crop extraction. Settings now expose the preparation worker limit, defaulting to available cores; effective concurrency also respects mapped source bytes, selected crops and encoder memory.

A read-only optimized Swift harness extracted three exact 2048 × 2048 crops from the existing Farm Furrows RGB16 8192 × 8192 normal map. The compressed source was 363,824,405 bytes; retained crop pixels were 75,497,472 bytes. Extraction took **5.27 seconds**, with **445,612,032 bytes maximum RSS**, **77,775,616 bytes macOS peak memory footprint**, and **zero swaps**. No original was modified. This measures crop extraction only; it does not measure complete import, preparation, export or model training, and it is not a throughput guarantee.

The exact command, captured measurements and standalone harness are saved locally in `out/native-dataset-fixes/stream-crop-benchmark.md` and `out/native-dataset-fixes/stream-crop-benchmark.swift`. Streaming regression fixtures cover all five filters, every Adam7 pass, RGB/gray/alpha channels at 8/16 bits, single-pixel dimensions, split IDAT boundaries, DirectX green conversion, invalid regions, corruption and cancellation. Target-scope regressions prepare height after unrelated normal/roughness originals have been removed, and verify that profile bytes remain in the PNG while old manifests compact without losing reviews.

## Build 106 validation

The final native suite passed **279 tests, zero failures and zero skips**, including the optional RAW-photo workflow using the existing `testimg/IMG_1951.DNG` fixture. Result: `build/TextureStudio/Logs/Test/Test-TextureStudio-2026.10.09_11-45-29--0400.xcresult`; log: `out/native-dataset-fixes/final-native-tests.log`. Build-tool and installer regressions passed as part of `make test-native`.

App-hosted interaction tests deliver mouse events to rendered subject/crop arrows, verify the selected row's visible highlight and the actual loaded image pixels, switch the map buttons below the image, and exercise Fit/Actual pixels. A separate native Settings control test changes the worker Stepper in both directions and checks persistence in a recreated store. External desktop automation was unavailable because the native computer-use connection could not start; these are in-process native window/control tests, not a claim that every application screen was manually exercised.

Optimized Release **0.9.13 build 106** was built, packaged and installed at `/Applications/Texture Studio.app`, including all four nested tools. Strict code-signature, native-bundle, version and installed smoke checks passed. Package SHA-256: `d4126c17fccfc6735c41be21c5c46038f81accea123675e9f02d54f042339dab`. The installed Dataset tool launched and remained running. An attempted external screenshot was blocked by missing macOS Screen Recording permission, so visual evidence remains the native hosted-window test captures. Build/install and smoke logs are in `out/native-dataset-fixes/`.

The reported 11:19 Debug-host stack overflow was traced to an observed worker setting assigning itself in its setter. The final setting uses a separately observed backing property and a clamped, immediately persisted setter. Worker tests cover extreme and invalid values. A later test-host interruption exited with code 0 without a crash report; the interrupted lifecycle test passed in isolation, and the entire final suite passed after removing unnecessary test-host foreground activation.

Prepared-image reuse tests now verify that unchanged files require no full-file digest reads, that missing or changed file state triggers one integrity check, and that changed image bytes are rejected. Displacement preparation accepts only diffuse plus displacement, including when unrelated original roughness/normal maps are absent. Legacy preparation with extra map roles cannot bypass regeneration for the chosen target.

## Complete native preparation and cache reuse measurement

A second optimized standalone harness created a fresh temporary dataset from the existing Farm Furrows 8K RGB16 diffuse and grayscale16 displacement pair, requesting height at 2048 pixels with one worker. Initial source registration took **0.867 seconds**. Complete first preparation—including crop decoding, lossless PNG encoding/writes, metadata, validation, publication and result loading—took **8.404 seconds**. The same unchanged request reused the prepared data in **0.003710 seconds**.

The result contained exactly **three samples with diffuse input plus height**, **six PNGs totaling 81,215,408 bytes**, and **13,285 bytes of sample metadata**. The entire harness, including initial registration and both requests, reached **428,539,904 bytes maximum RSS (408.69 MiB)** and **311,346,088 bytes macOS peak memory footprint**, with **zero swaps**. These process-wide high-water values are distinct from the phase timings. Source file states remained unchanged, and the fresh temporary dataset was removed; no original or existing user dataset was modified. This measures native preparation and reuse, not model training, inference or general throughput.

Commands, detailed scope and measured output are saved locally in `out/native-dataset-fixes/full-preparation-benchmark.md`, `full-preparation-benchmark.swift`, `full-preparation-benchmark.json` and `full-preparation-benchmark-resource.log`.
