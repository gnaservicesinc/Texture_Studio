# Fresh bark preparation and rectangular source crops

Texture Studio 0.9.23 (118) selects a native square field before selecting the training crop in a source whose long edge is at least twice its short edge. A 2048 × 4096 source therefore offers two 2048-square fields. At a 1024 training grid, the existing center-crop policy applies inside the selected field. The folder and epoch seed choose the field; paired diffuse, height, normal, and roughness maps share the recorded coordinates. Each source resolution still contributes one spatial training crop.

Quarter-turn augmentation now rotates the recorded rectangular-source crop. It cannot jump back to the center of the entire tall source. Complete rectangular input/target pairs also satisfy source preparation without downloading an absent smaller nominal tier; genuinely missing targets are still repaired.

`--fresh-training --dataset PATH` explicitly opens an RGB compact height model with warm start disabled, all materials included, no old checkpoint selection, and no previous comparison or output selection. Optional `--training-size`, `--learning-rate`, and `--updates-per-map` values are validated before settings change. Opening this state does not start training. A matching specialist binding retains the exact dataset membership needed to register a subsequently completed model.

## Validation

- 81 focused tests: 80 passed, one opt-in cache test skipped, zero failures. Coverage includes portrait and landscape source fields, deterministic seeds, balanced material weighting, paired native integer codes, recorded-crop quarter turns, missing-target recovery, fresh settings, and invalid launch arguments without mutation.
- Release build and packaging passed, including native bundle/payload and nested signature checks. Native build-tool tests passed.
- Installed 0.9.23 (118) in `/Applications/Texture Studio.app`; built and installed bundles compare byte for byte; deep strict signing verification and the installed native smoke test passed.

## Authorized local dataset preparation

The opt-in `BarkFreshPreparationProductionTests` maintenance case uses the verified source inventory outside the repository. It imports 18 distinct bark families and 51 paired resolution sets from `/opt/ipde/sources_mats`. The title-case Pine Bark folder duplicates `pine_bark` byte for byte and is omitted from membership; both original folders remain intact.

Preparation verifies 51 training crops and one extra known-subject diagnostic crop at 1024 × 1024. Every diffuse and 16-bit height crop is compared against its exact native source rectangle; all 221 original PNG hashes are checked before and after preparation. Temporary verification crops are cleaned before publication, leaving no old or newly prepared training cache in the fresh dataset.

The previous Bark dataset, its caches, previous training/comparison output directory, and selected checkpoint activation files are retained in an archive outside the active workspace. The Bark specialist model list and saved checkpoint selections are cleared. Trainer opens with learning rate 0.0001 and 100 updates per map, without DepthPro or an automatic training start. These are execution and data-fidelity checks; they do not establish quality or generalization of a future full training run.

Local receipts and detailed logs are in:

`~/Library/Application Support/Texture Studio/Validation/bark-fresh-start-2026-10-12/`

The maintenance test is intentionally opt-in and refuses to repeat publication over an existing archive:

```sh
TEST_RUNNER_TEXTURE_STUDIO_BARK_FRESH_SETUP=1 xcodebuild \
  -project src/TextureStudio/TextureStudio.xcodeproj -scheme TextureStudio \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build/ScalarAgreementTests ENABLE_TESTABILITY=YES \
  ENABLE_HARDENED_RUNTIME=NO -parallel-testing-enabled NO \
  -only-testing:TextureStudioTests/BarkFreshPreparationProductionTests test
```
