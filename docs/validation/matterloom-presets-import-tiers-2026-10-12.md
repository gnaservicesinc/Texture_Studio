# MatterLoom presets, imports and source tier limits

The goal is a highly detailed, visually pleasing texture that passes practical sanity checks. Professionally artist-authored 16-bit material maps are valid clean targets. This change supplies model capacity controls, an optimizer option, importer fixes and better metadata; it does not establish learned texture quality or ship newly trained photo cleanup/detailer weights.

## Changes

- MatterLoom is the pipeline name. Hugging Face default repository slugs now include the recorded model name, map target and checkpoint hash. Exported model cards describe the named specialist, configured training subjects, original dataset name and actual saved settings. Selected-subject preparation keeps the original display name even when its temporary manifest has no name.
- Fresh compact model presets use widths 16/22/32/46/64: 256,257 / 478,017 / 999,937 / 2,050,497 / 3,949,569 RGB-to-scalar parameters. Counts for normal and DepthPro models are derived from their actual shapes. Existing checkpoints retain their recorded widths.
- Lion implements the published sign/momentum ordering and decoupled weight decay, with one momentum value per parameter. AdamW remains the default. The UI offers explicit lower-learning-rate starting settings; optimizer choice and settings are recorded. Lion's small optimizer-state saving does not remove large activation/buffer costs at higher grids.
- New Dataset and Dataset Info offer directly editable source tier counts from 1 to 3, defaulting to 1. A 2K training grid fetches only 2K by default, with 4K and 8K explicit options. Small training grids start recovery at the smallest published 1K tier. Existing local higher-resolution originals stay discoverable and byte-preserved; preparation still writes only crops at the selected grid.
- Import recognizes common explicit diffuse/base-color, displacement/height, normal and roughness synonyms plus authored pixel suffixes. FabricSubstance003 `_515` maps are paired by their actual 512×512 dimensions. Ambiguous duplicates still require selection, and packed or semantically different maps are not silently relabeled.

## Verification

These are separate focused runs with overlapping test cases; their totals must not be added as a unique-test count.

- Final current-source integration: 50 tests executed, 3 opt-in skips, zero failures. Covers upload naming, cards/package bindings, native compact training, missing-source recovery and native trainer completion/cancellation/optimizer settings.
- Dataset/tier run: 79 tests passed; after the small-grid tier correction, 5 selected current-source cases passed again, covering 256/512 recovery, 2K tier limits, saved counts, scan/import/preparation and sheet rendering.
- Model-size/optimizer/comparison work: 60 unique focused cases executed, including 3 existing opt-in skips, with zero failures; additional prepared-name/subject metadata regression passed.
- Import: 12 tests passed, including real offline previews of FabricSubstance003 (4 sets), Pine Bark (3 sets) and pine_bark (3 sets), no warnings or ignored PNGs. SHA-256 checks verified all 40 original maps remained unchanged. See `matterloom-import-proof-2026-10-12.json`.
- Native build-tool tests passed. Final Release build/package passed. Native payload verification and deep strict signatures passed for the parent and bundled tools. Installed app version remains 0.9.23 (118), as a non-release snapshot. Built/installed bundles compare byte for byte; installed smoke printed `Texture Studio native smoke passed`.

One initial broad run compiled while the small-grid tier fix was still being integrated and failed the new small-grid assertions. The corrected current-source suite passes. That broad run also overlapped the 2K capacity probe and encountered GPU memory pressure in an unchanged PBRnxt numerical test; it was stopped. GPU execution probes were subsequently separated into fresh processes. No full-suite pass is claimed.

## Native dimension and capacity probes

Apple M2 Max, 64 GiB unified memory, native Float32, reduced-precision fast math disabled. Inputs/targets are patterned numeric fixtures; they verify graph execution at the declared grid, not learned visual quality. Debug timings include cold compilation and do not represent warmed Release throughput. CPU RSS excludes much Metal/GPU accounting. Metal allocation figures are current allocations at completion, not sampled peaks.

| Parameters | Operation | Native grid | Result | Measured duration |
| --- | --- | --- | --- | --- |
| 478,017 | One backward/AdamW update and post-update forward | 2048×2048 | Passed; changed weights and finite numerics | 15.08 s model run |
| 2,050,497 | One backward/AdamW update and post-update forward, fresh process | 1024×1024 | Passed; changed weights and finite numerics | 6.84 s model run |
| 2,050,497 | Inference only, fresh process | 2048×2048 | Passed; exact output dimensions and finite values | 2.94 s forward |
| 2,050,497 | Training, fresh isolated process | 2048×2048 | Actual Metal out-of-memory during backward; stopped, no update | No completed update |

The half-million 2K probe reported 47,553,134,592 bytes of Metal allocations at forward completion. The two-million 1K probe reported 24,750,080,000 bytes; its 2K inference probe reported 20,336,672,768 bytes. The failed training command returned `kIOGPUCommandBufferCallbackErrorOutOfMemory`. No predictive memory veto was added. The smaller model demonstrates that 2K training can execute on this system in a bounded probe; a long stable run remains unqualified.

Receipts: `matterloom-2048-half-million-probe-2026-10-12.json`, `matterloom-1024-two-million-probe-2026-10-12.json`, `matterloom-2048-two-million-inference-2026-10-12.json`.

Detailed logs remain outside the repository in `/tmp/matterloom-final-integration-tests.log`, `/tmp/texture-studio-tier-tests.log`, `/tmp/texture-studio-tier-final.log`, `/tmp/texture-import-synonyms-tests.log`, `/tmp/texture-model-presets-tests.log`, `/tmp/texture-model-presets-256-tests.log`, `/tmp/texture-model-presets-comparison-tests.log`, `/tmp/texture-prepared-model-card-tests.log`, `/tmp/texture-compact-2048-probe-test.log`, `/tmp/texture-compact-2048-width46-training.log`, `/tmp/texture-compact-2048-width46-inference.log`, `/tmp/texture-compact-1024-width46-training.log`, and `/tmp/matterloom-final-package.log`.

The proposed learned photo-to-diffuse and final restoration stages, including rendered artist-material training pairs and practical quality criteria, are described in `docs/matterloom-pipeline.md`.
