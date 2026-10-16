# Dataset tier setup validation

The reported PolyMat setup contained 248 registered resolution sets in 123 source folders. It had only three 1K sets, was missing 120 requested 1K tiers and 11 next-level 2K tiers, and already contained ten 8K fields. The previous balanced crop selector emitted one crop per available resolution correctly, but setup did not acquire and register the missing tiers before the import preview. Its displayed inventory therefore could not meet the requested three-tier plan.

Setup now recovers published, contract-compatible 1×, 2× and 4× tiers before a folder scan, direct import, original dataset read or training-size edit. It maintains all target maps even when height is selected for preview. Proof-based import uses the completed preview without initiating additional recovery. Existing imported folders outside the managed dataset are included, with recovery scoped to their registered provider families. New diffuse tiers above 4× are not fetched; only missing targets for existing higher diffuse fields are eligible. This is native Swift, including ambientCG ZIP extraction, with no additional interpreter.

Poly Haven and ambientCG names select their respective provider. Both retain the full published PNG and its exact numeric codes. Numbered Poly Haven color roles carry forward the lowest compatible existing variant into a missing input tier: published `coll1` and normalized `col_1` identify the same color role, and the recovered input remains bound to native variant `color_1`. Existing additional variants retain their own source bindings. Source-directory aliases are grouped by their canonical location so each missing map is requested once.

A rectangular field can identify a nominal provider tier when either original axis matches its filename's resolution within 2%; actual crop eligibility still requires enough pixels on both axes. Therefore three nominal resolution sets do not necessarily provide three usable crops. Missing displacement remains unavailable when the provider does not publish it or it fails the 16-bit contract. Downloaded originals have a source-bound hidden proof sidecar, including archive/member evidence for ambientCG.

The UI passes the chosen grid into setup, refreshes the scan if that grid changes before import, and binds import to the post-recovery index and plan hashes. Prepared-dataset management checks its durable source snapshot before cleanup and uses a read without recovery to retain that proof. Dataset labels distinguish source folders from resolution sets; Trainer counts distinguish the selected folder from the complete plan. Setup progress and unavailable-map reasons remain visible.

Automatic recovery and discovery also honor fully removed families in owned source storage. A larger training grid cannot restore one through an untombstoned new tier, and unrelated folder imports cannot start downloads for it. Fresh source families still appear, and explicit import of the removed family's folder can restore it. The regression covers both new scoped tombstones and legacy entries with opaque source-set IDs, with original bytes unchanged.

The [first-pass receipt](dataset-balanced-tier-setup-2026-10-11.json) records expansion from 248 to 375 resolution sets, with 518 recovered maps and no failed downloads. Its ten unavailable maps exposed the numbered-color aliases and the cloth exceptions below. The [final receipt](dataset-balanced-tier-final-2026-10-11.json) records the completed cleanup and two recovered leather inputs. The opt-in setup test performs no training, records every folder's nominal tiers and usable dimensions, preserves the material IDs and reviews present when that test starts, and rehashes every previously recorded retained source map and color variant. A metadata snapshot from before this repair is retained locally under `out/validation/polymat-before-build111`.

The user authorized removing six problem families from this local public corpus and its PolyMat records: Fabric Pattern 05, Fabric Pattern 07, `jersey_melange`, `scuba_suede`, `waffle_pique_cotton` and `polar_fleece`. Two lack published height maps; four have undersized 1K fields, with two also having mismatched 2K target rows. The [cleanup receipt](dataset-cloth-corpus-cleanup-2026-10-11.json) records deletion of all six original folders and removal of their 16 resolution sets. It applies no pixel corrections or resizing. The shipped app's ordinary setup preserves existing user source files and adds missing published originals; it does not perform this corpus deletion automatically.

| Final post-cleanup audit | Result |
| --- | --- |
| Source folders | 117 |
| Registered resolution sets | 361 |
| Folders with all required nominal tiers | 117 of 117 |
| Valid height training crops at 1024 | 361 |
| Spatial crops per folder | 107 folders have 3; 10 folders have 4 |
| Final recovery | 2 inputs recovered; 0 unavailable; 0 failed |
| Retained source hashes and reviews | Unchanged |

Every remaining folder supplies valid 1K, 2K and 4K paired height fields; ten retained 8K fields supply the fourth crop. An independent audit of all 1,444 primary PNG maps found no missing files, dimension mismatches or unusable height tiers, and confirmed all six deleted folders are absent. The final opt-in setup test passed in 106.148 seconds. No material or refiner training was started.

## Build and checks

The final Release suite executed 138 tests: 136 passed, with the two opt-in corpus tests skipped and no failures. Both corpus tests were run separately and passed: source-folder cleanup and the final retained-dataset setup audit. The removed-family regression passed in the final suite.

Texture Studio 0.9.16, build 111, was built with `make build`, packaged with native bundle and signature validation, and installed using the Swift installer with `--if-closed`. The installed bundle matches the built bundle byte for byte, passes deep strict signature verification, and passes the native smoke test. The built app also passed its smoke test. Interactive UI automation was unavailable because the native computer-use pipe failed to start; the UI behavior is covered by focused store and control tests.

Package SHA-256: `8ca8f4d838a7710df71a93ded914eaa96265199d54af2d3922ad97a5262105cf`.

## Reproduction

Run the focused native test classes `NativeMaterialDatasetTests`, `NativeMaterialSourceRecoveryTests`, `NativeMaterialCropRecoveryTests`, `DatasetManagementTests`, `WorkbenchPreferencesTests`, `WorkbenchTrainingCompletionTests` and `WorkbenchTrainingProgressTests` with the TextureStudio scheme. The Release configuration avoids the substantial Debug cost of inspecting large numeric PNGs. Enable testability and disable the hardened runtime for this local test build. The two production-corpus tests skip in an ordinary suite unless their separate opt-in gates are enabled.

The retained-dataset test is `NativeMaterialDatasetSetupProductionTests/testRealDatasetSetupRegistersBalancedTiersBeforeTraining`. Enable it through the Xcode test runner environment:

```sh
mkdir -p out/validation

TEST_RUNNER_TEXTURE_STUDIO_DATASET_SETUP_VERIFY=1 \
TEST_RUNNER_TEXTURE_STUDIO_DATASET_SETUP_PATH='/path/to/user/Documents/Texture Studio Datasets/PolyMat' \
TEST_RUNNER_TEXTURE_STUDIO_DATASET_SETUP_SIZE=1024 \
TEST_RUNNER_TEXTURE_STUDIO_DATASET_SETUP_REPORT=/path/to/workspace/ipde/out/validation/dataset-tier-recheck.json \
xcodebuild -project src/TextureStudio/TextureStudio.xcodeproj -scheme TextureStudio \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build/DatasetTierSetup111 \
  ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO \
  -only-testing:TextureStudioTests/NativeMaterialDatasetSetupProductionTests/testRealDatasetSetupRegistersBalancedTiersBeforeTraining test
```

This test deliberately performs source setup and manifest registration. It does not start refiner or material training. The separate recheck path preserves the retained first-pass receipt.

Physical deletion of the six authorized families is confined to `NativeMaterialDatasetSetupProductionTests/testRemoveUnusableClothFromAuthorizedDevelopmentCorpus`, gated by `TEXTURE_STUDIO_DATASET_CORPUS_CLEANUP=1` (forwarded as `TEST_RUNNER_TEXTURE_STUDIO_DATASET_CORPUS_CLEANUP=1` through Xcode). That method is excluded from the command above and ordinary app setup.
