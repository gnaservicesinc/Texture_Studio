# Texture surface validation — 2026-10-07

This record covers the DA3-GIANT-1.1 and material-relief corrections requested after the original Swift migration. It supersedes the original validation record's Small V2, embedded-depth and photographic-height paths. Original photos and existing external model folders were preserved.

## Environment and selected model

- Apple M2 Max, 64 GiB unified memory; macOS 27.2 and Xcode 27.2.
- Managed Python 3.14.8, PyTorch 2.14.0, Float32 inference on MPS. Dependency versions and Apache-licensed upstream inference source are bundled in the app; Python and weights are separately managed and removable.
- Exact model: `depth-anything/DA3-GIANT-1.1`, revision `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19`.
- Checkpoint SHA-256: `1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001`; 5,422,814,644 bytes. Config SHA-256: `74626a50d6dee2a11820291a4305c1a34aa5adc4f7260908bbdbc9a939ba8e93`.
- Upstream inference revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. The requested checkpoint's published license is **CC BY-NC 4.0**. Code licensing does not change the weight license. [Official model card](https://huggingface.co/depth-anything/DA3-GIANT-1.1).

The app supports the PyTorch/MPS DA3 backend and optional custom Core ML models. The Apple Small V2 model is absent from the active catalog. Its registered app-managed download was safely retired; external model files were not deleted. Missing weights and runtime paths lead to Locate/Download/Install recovery rather than a silent substitute.

## Numerical checks

The optimized inference path evaluates the same primary depth head while omitting independent camera/ray/Gaussian outputs. A real 392×294 Float32 MPS comparison against the upstream complete head was bitwise equal, with maximum error zero. The result is recorded in `out/da3-validation/primary-parity.json`.

Raw model Float32 samples are separate from the artistic height conversion. The material pipeline removes a robust global plane, cleans isolated depth spikes and applies bounded edge-aware filtering. Near-flat protection reduces amplification of very small positive-distance variations; it is an artistic safeguard, not a physical confidence estimate. Brightness-derived relief defaults to zero. Embedded portrait/effects depth cannot supply material relief, including through old recipes.

Analytic tests cover plane removal, retained broad relief, isolated-spike cleanup, preserved step edges, higher-is-raised attached height maps, camera-distance inversion, neutral maps without depth, matching image/depth orientation, and OpenGL +Y normal direction. The isolated analytic Metal normal ramp differed by at most approximately 2.98e-8. The full-engine regression also verifies normals against the final height gradients within 1e-6.

Visual validation exposed a depth-resampling border bug: zero/transparent samples outside a low-resolution depth image created a false ridge after upscaling. Numeric resampling now clamps valid source texels before resizing. The regression checks a 32-pixel depth grid against a 512-pixel photo and 1024-pixel export: the flat border remains exactly 0.5 height and (0.5, 0.5, 1) normal while interior peaks remain.

## Supplied photographs and actual exports

Representative wicker (`IMG_1935.HEIC`, `IMG_1942.HEIC`) and painted-wall (`IMG_1947.HEIC`, `IMG_1949.DNG`) photographs from `/opt/ipde/testimg` were processed with the exact GIANT checkpoint. Native engine exports and display derivatives are under `out/da3-validation`; the final contact sheet is `out/da3-validation/material-contact-sheet.png`.

All **16 supplied files** also passed actual native import and bounded-preview checks: eight HEIC and eight DNG, with EXIF orientation 3 applied. HEIC source dimensions were 5712×4284 or 4032×3024; DNG dimensions were 4032×3024 or 8064×6048. The durable import report is `out/da3-validation/native-imports-16.json`. This checks import/orientation for the whole folder; it does not claim DA3 inference on every file.

The final Release app's actual `TextureWorkspace` import → DA3 inference → preview → cached export path passed on wicker HEIC, wall HEIC and wall DNG. The corrected final exports are `out/da3-validation/store-exports/IMG_1935`, `IMG_1947` and `IMG_1949`. Each 1024×1024 material folder contains 8-bit sRGB diffuse PNG, linear Float32 roughness/normal/displacement EXRs, Blender setup files and `depth-source.json` with the selected checkpoint, backend, device, dimensions and precision. Actual EXR channel types and map dimensions were inspected after writing.

At the explicit 1036-pixel inference edge, the final model runs took 10.30, 10.65 and 10.42 seconds respectively; the complete import/preview/export validation took 20.02, 17.17 and 31.39 seconds. A 1540-pixel wall inference took about 15 seconds. One 1036 run reported approximately 11.24 GB peak child resident memory and 9.79 GB Metal driver allocation; the 1540 run reported approximately 14.84 GB Metal driver allocation. These measurements overlap and must not be added together. Input size is explicitly selected and independently checked against available memory; export size does not imply transformer inference at that size.

## Blender verification and observed quality

Blender 5.3.0 Alpha loaded all three corrected native-app exports headlessly. Diffuse loaded as sRGB; all three EXRs loaded as float Non-Color data. The generated script's default has normal strength 1 and geometric displacement disconnected. With its geometric-displacement option enabled, displacement is connected and equivalent normal strength is zero. This prevents applying the same height twice.

Six Cycles CPU renders at 512×512 with 16 samples under neutral and grazing lights are saved as `out/da3-validation/blender-{wicker,wall,wall-dng}-{neutral,grazing}.png`; the audit is `out/da3-validation/blender-validation.json`. These are visual review renders, not high-sample production renders or verification against a stable Blender release.

Wicker predictions contain useful raised strips and recesses aligned to the photograph. Painted-wall predictions mostly contain broad, low-contrast depth patches; the DNG render also shows weak structured variation that is not established surface grain. Near-flat protection restrains those artifacts; increasing inference resolution did not establish reliable fine paint-grain height. DA3 is a scene-geometry prior, not a validated material-detail estimator. Neither 32-bit storage nor a larger export creates missing detail. Hard-shadow reconstruction, trained material roughness and seamless tiling remain unfinished quality work. See [material refinement research](material-refinement-research.md) for released alternatives, license limits and a Poly Haven/MatSynth training plan.

## Build and regression evidence

- Final hosted XCTest suite: **63 tests passed, zero failures**. Engine 18; model lifecycle 16; Python depth service 7; decision service 10; photo evidence 4; spatial fusion 3; workspace 5. Log: `/tmp/texture-studio-da3-native-63.log`; result bundle: `build/TextureStudio/Logs/Test/Test-TextureStudio-2026.10.07_08-00-46--0400.xcresult`.
- The real runtime setup helper passed a fresh managed install → MPS probe → safe removal. An offline cancellation fixture with a child that ignored SIGTERM exited with status 130 in approximately 1.02 seconds and reaped the child. This exposed and fixed waiting inside a Python signal handler; cancellation now signals, polls and reaps the child process group. Audits are `out/da3-validation/runtime-cancellation.json` and `out/da3-validation/runtime-install-removal.json`. The separately installed user runtime and GIANT weights remain available.
- Final Release build and packaging succeeded, including signature verification and archive integrity. The archive is `dist/Texture-Studio-macos-arm64.zip`, SHA-256 `664a86817b5966e0245539dd5c31d31a1e33c2f6169eb308bfe34d180a762685`. Bundled `setup_runtime.py` matches the validated source SHA-256 `28d3409c88235caebce8b1291b07231a8da02647440fe231c0b5e09abc230896`.
- The canonical `./script/build_and_run.sh --verify -- --open /opt/ipde/testimg/IMG_1935.HEIC` passed and left the rebuilt Debug app running with that photo. Log: `/tmp/texture-studio-da3-final-run-verify.log`. `git diff --check` and the repository version check passed.

This record does not claim CI, notarization, public distribution or trained-model quality. Interactive screenshot/accessibility QA was unavailable; actual inference, export, numeric inspection and Blender rendering were used instead.
