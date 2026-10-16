# DepthPro resource recovery and user output storage

Texture Studio 0.9.17, build 112, resolves a missing DepthPro selection before constructing its numeric cache. Native training, checkpoint comparison and Studio inference share this acquisition path. No interpreter or conversion runtime was added.

The compatible Core ML package is pinned to `coreml-projects/DepthPro-coreml` revision `e3b374e875fa7c89ca4c80203e1fb12e5e02bceb`. The installer checks the published byte count and SHA-256 of all three package files, then verifies complete package identity `e62c68496d25c76e099fbbcfb7a39814a201a1f2d9a6a9411e37feec2d5d94ec`. It publishes the 1,902,515,337-byte package in `~/Library/Application Support/Texture Studio/DepthPro`, retains verified staging members for retries, repairs incomplete managed packages, and serializes concurrent installers. Existing developer-selected models keep their own identity. Missing selections never create a package at their stale path.

Native download tasks report intermediate bytes through their download delegate. Cancellation before task registration is remembered; registered cancellation waits for the serialized delegate's file operation to finish before caller cleanup. Package publication checks cancellation before replacing the active directory.

The default material workspace is `~/Library/Application Support/Texture Studio/Material Workspace`. Legacy preferences inside a Git checkout, including nested paths, worktrees and symlink aliases, migrate before new output is created. Missing repository model-download locations also migrate. Existing selected developer weights and completed runs retain their paths. The previously empty `/path/to/workspace/ipde/out/material-training` directory was removed with `rmdir`.

## Genuine model execution

The [cold acquisition receipt](depthpro-acquisition-native-2026-10-11.json) records the successful native missing-model recovery. It acquired the genuine package, cached a DepthPro prediction, completed one optimizer update, exported the checkpoint, and ran inference from those saved weights. The complete acquisition, training and inference check took 141.04 seconds.

The [final-source reuse receipt](depthpro-acquisition-reuse-native-2026-10-11.json) repeats missing-selection recovery, one training update and checkpoint inference against the installed package. It completed in 6.37 seconds with zero model download events.

The pair was an exact aligned 256×256 center crop at `[896, 896, 256, 256]` from `white_stucco_02`'s original 2K diffuse and displacement maps. The width-16 refiner had 255,969 parameters, one input channel and one height output channel. One update completed with no skipped samples. Inference produced matching 256×256 Float32 EXR output and recorded the pinned upstream package identity. This verifies execution and recovery; it does not qualify trained detail, novel-material generalization or 1024 training quality.

Original files were hashed before and after the check and matched:

| Original | SHA-256 |
| --- | --- |
| `white_stucco_02_diff_2k.png` | `c1d34a64fa62b613e62d24a81c20e9d1f612e555525a1476742fea6b98e563ac` |
| `white_stucco_02_disp_2k.png` | `1ff58140eb77ad6fd1350f30952eae6625a66041317e6c99ab64faf42d4ca82d` |

## Regression coverage

Focused native checks cover missing selections, offline reuse, managed-package repair, verified staging reuse and interrupted-transfer cleanup, model aliases, concurrent installation, corrupt downloads, retries, cancellation, the pinned inventory, byte progress, ready-path persistence, workspace migration, training dispatch, compact and DepthPro model selection, checkpoint comparison, cache alignment, native routing and generic model management. Opt-in HTTPS checks exercise intermediate download callbacks and cancellation during a real published artifact transfer.

The project [working principles](../../AGENTS.md) require attempting available authorized recovery paths before reporting an error, including obtaining and validating missing public resources.

The final optimized native test batch passed: 124 tests, two separate opt-in checks skipped, zero failures. The live acquisition/reuse, published-artifact progress and native cancellation checks were enabled and passed. Release build, native build-tool checks, version consistency, packaging, installation and installed native smoke also passed. The installed signed app matched the staged bundle byte for byte and passed `codesign --verify --deep --strict`. The installed Material Trainer launched, persisted the user Application Support workspace, and did not recreate the repository's `out/material-training` directory.
