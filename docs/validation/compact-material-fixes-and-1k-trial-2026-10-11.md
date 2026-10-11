# Compact material fixes and 1K trial — October 11, 2026

Texture Studio 0.9.14, build 109, fixes imported checkpoint comparisons, native model loading, the final cleanup error and recognition of an existing Hugging Face CLI login. At the time of this build 109 assessment on 2026-10-11, the depth-only refiner remained an idea for a separate experiment; this assessed change did not implement or select that pipeline.

## Findings and fixes

- Imported-photo comparisons copied the 512 training grid into diffuse preparation, whose validator accepted only material export sizes starting at 1024. Preparation now accepts 256 and 512 as well as the export grids. One prepared 16-bit input is shared across compared models; original photos stay unchanged.
- Photo, supporting-frame and attached-map imports no longer reject sources above an arbitrary 150-megapixel limit. Finite integral dimensions and allocation-overflow guards remain. Source dimensions are independent of the requested prepared/output grid; an actual platform decode or resource failure can still be reported.
- Local Models now offers **Locate Material Model…** for native material safetensors and export folders. It validates the complete checkpoint and activates its exact digest. The developer Core ML action identifies and filters its own format, and gives an actionable message for native checkpoints.
- The user's run completed all 6,425 steps over 25 epochs and published `export/model.safetensors`. Its prepared manifest SHA-256 matches the run's recorded dataset digest (`32f23014579d27209a1232ff1c52176af1e97c2940a9c9bcba1e96ce5d00dac6`). Its only undeclared cache file was Finder's `.DS_Store`, which triggered the cleanup guard. Cleanup now recognizes regular Finder metadata by exact filename and version/Bud1 signature. Unrelated files, misleading filenames, symlinks and unrelated folders remain protected.
- Once final weights are saved, library refresh, source-dataset restoration, cleanup and optional upload failures leave training completed with a descriptive notice and retained package path. Cancellation arriving after the saved final response also preserves that completed result. Worker exceptions are now written to operation logs.
- The app reads the CLI's active token file using `HF_TOKEN_PATH`, `HF_HOME`, `XDG_CACHE_HOME` or the default cache, alongside environment and app Keychain credentials. It never launches Python or the HF CLI for runtime authentication. The saved CLI login is read-only. A live native whoami test recognizes `alastandy`; no model was uploaded as validation.
- A verified remote upload receipt remains available if local catalog registration or model-list refresh fails. Those later failures are reported separately from publication.

## 512 and 1024 throughput

Short scalar training probes on the same Apple M2 Max use width 32, 999,937 parameters, three distinct procedural RGB/16-bit-height pairs, two epochs and six optimizer steps. Both native grids are prepared directly; the trainer does not resize them. These probes do not evaluate model quality.

| Native grid | Mean of five warmed steps | First step | Entire six-step run | Process peak resident memory | Metal allocation at completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 × 512 | 0.224858 s | 0.460522 s | 1.709507 s | 381,501,440 bytes | 424,214,528 bytes |
| 1024 × 1024 | 0.697660 s | 1.327946 s | 4.950386 s | 839,729,152 bytes | 1,657,405,440 bytes |

The warmed 1K step is approximately 3.10 times the 512 step. First-step shader/graph caches differ between probes, so compare warmed timings. Resident memory is process rusage, not the full unified-memory footprint; the Metal completion allocation is not a peak. These figures overlap and must not be added.

A 1K trial is practical. The current compact architecture already supports 1024 without changing its width or parameter count. Continue the same target's model at 1024 or start a separate run, using original maps that supply that native size. More pixels provide finer supervision, but increasing resolution alone does not prove that learned blur or incorrect relief improves. Compare an unseen material subject at the same resolution and keep model identity and source preparation fixed.

A separate six-step probe loaded the user's actual step-6425 height checkpoint and continued it at 1024 to step 6431. The new temporary export records `training_size: 1024`; the source checkpoint SHA-256 remains `127c8856f78ccec6c035eb3816b3cfce35447645389736721445ab835af09b48`. Its five warmed steps average 0.711757 seconds. The probe's temporary training/export files were removed; no replacement was made to the saved user model. Its receipt is `compact-native-1024-continuation-benchmark-2026-10-11.json`.

Raw timing receipts are `compact-native-512-benchmark-2026-10-11.json` and `compact-native-1024-benchmark-2026-10-11.json` beside this report.

## Depth-only refiner assessment

This historical assessment describes the proposal before the separate [DepthPro Height Refiner implementation](../depthpro-height-refiner.md). Its no-download and no-training statements apply to this assessment only.

The proposed training pair is **DepthPro-generated depth as one input channel** and **original material height as one target channel**. The original diffuse photo is used to generate the DepthPro map beforehand, not as a refiner input. Production would run DepthPro once and then the one-channel refiner. Cached DepthPro predictions would avoid running the large model inside every training epoch.

This is a reasonable restoration experiment: a small network could learn systematic depth errors, recurring shapes and boundary corrections with less work than learning shape entirely from scratch. Detail completely absent from the input remains ambiguous; the network may infer plausible rather than correct fine relief. Whole-material holdouts are necessary to judge whether the added detail is useful.

Apple's [DepthPro reference implementation](https://github.com/apple-aiml-research/ml-depth-pro) predicts camera depth in meters. That differs from a material's relative surface height, so the experiment needs consistent direction, scale and coordinate preparation across cached training input and production inference. Preserve float data and paired spatial alignment rather than treating display grayscale as the numeric target contract. One channel reduces input/output work, but the internal feature layers dominate the current compact network, so this change alone should not be expected to provide a threefold speedup.

A [Core ML DepthPro conversion](https://huggingface.co/coreml-projects/DepthPro-coreml) exists with an RGB image and required `originalWidth` input. Texture Studio's current developer Core ML importer accepts one required image input, so that conversion would need a dedicated native input adapter and measured hardware validation. It is not automatically compatible with the present generic depth-model loader. No DepthPro weights were downloaded or new refiner training started for this assessment.

## Validation

- Full optimized Release suite: **425 tests, six optional skips, zero failures**, 556.805 seconds. The live HF identity test and actual user-checkpoint production test were enabled and passed. Result bundle: `build/TransparencyTrainingValidation/Logs/Test/Test-TextureStudio-2026.10.11_02-09-00--0400.xcresult`.
- The actual step-6425 checkpoint was imported with a temporary selection registry, applied to a newly generated 2048×2048 procedural RGB PNG through the production 512 preparation/inference paths, and verified as finite scalar Float32 EXR. Production numeric readback matched bit for bit; the model and source bytes were unchanged. The fixture establishes functionality, not learned material accuracy. Receipt: `compact-production-verification-2026-10-11.json`.
- Focused completion/dataset/transfer/upload batch: 51 tests, zero failures. The earlier 1K/import/auth batch exposed four missing-field errors in new test fixtures; those fixtures were corrected before the successful full run. The individual 1K timing probe passed, and the final actual-checkpoint 1K continuation probe passed after adding its optional harness support.
- `make package`, native bundle validation, nested signature verification, native build-tool tests and source release-version checks passed. Runtime authentication and material inference/training remain native, with no Python interpreter bundled.
- Installed **0.9.14 build 109** using the native Swift installer with `--if-closed`. Recursive byte comparison against the verified release bundle is exact. Installed signature verification passed; installed `--smoke-test` prints **Texture Studio native smoke passed**. This is compiled/runtime validation; no live UI click-through is claimed.
- Main executable SHA-256, both built and installed: `90a7bd61b6f6aea09fc00da6720136ca80ce44443bcd6d6d72b72216cb36b66c`. Distribution zip SHA-256: `91dc03e89dd89ebe90f507748b34c5511f19956ceb6595761a5d972f693e2766`.
