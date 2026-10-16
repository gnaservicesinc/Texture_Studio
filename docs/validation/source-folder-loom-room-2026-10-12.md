# Source folders and Loom Room

Texture Studio 0.9.25, build 120, 2026-10-12.

## Behavior

Loom Room is the developer training starting point for MatterLoom. It shows original folders and image tags, links to classifier teaching, and guides category-specific height, roughness and normal training and review. Returning from category training or comparison restores the original source root. Adding images and preparing classifier examples therefore use the original folder displayed by the workshop.

A bare folder is the dataset. Creating one creates only a directory; listing one reads current membership, filenames, optional tags, file statistics and image headers. It neither requires source-root JSON nor invokes a model or hashes every original. Directory events trigger debounced background refreshes. The app can copy images/folders in, edit tags, rename a material folder, assign unclear maps and move a folder out. Finder is optional. Loom Room and Dataset startup also defer saved-checkpoint inspection. Generic Trainer, Compare and Saved Models actions reconnect saved files when needed; category workflows inspect only their selected model. Deferred paths and selections are retained, including when another category model is saved.

Optional extended attributes record image roles, teaching labels and verified source identity. Copies retain teaching decisions with their own identities and require their own verification. Replaced image bytes invalidate cached checks. Full verification and crop preparation happen when training requires them, with generated data in app-managed storage. Original image bytes are preserved.

The bundled classifier is the user's DTTC3 export, byte-for-byte identical to `Classifiers/diffuse_texture_type/DTTC3.mlmodel`, with SHA-256 `4971a8abfe6a7169b0ebb75e5462c1578557570dc885b28a85dee9a54f914086`. Core ML compilation, model loading and Vision compatibility succeeded without predictions. Its 360 × 360 image interface has 15 string classes and class probabilities, including Drywall, Fabric and Leather. The changed model identity invalidates cached automatic predictions; user teaching labels retain priority. This verifies compatibility, not its accuracy improvement.

Explicitly opening or creating another source folder selects that folder and its grid for classifier examples, clearing a stale external 1K override. An override chosen explicitly afterward remains supported. Internal category navigation preserves the user's current setting.

New folders default to 2048. Undersized originals remain inspectable; they cannot appear as 2048 training crops or classifier examples. Slight edge differences use the common native intersection without scaling. Height crops retain the original 16-bit integer codes and convert to model samples using 65535 as the denominator. AO and packed supplemental maps remain additional maps and cannot supply diffuse input.

## Corpus evidence

The real `/path/to/workspace/sources_mats` folder snapshot accounted for 1,897 PNG files and 462 native paired sets, with 324 exact 2048 crop views. The final recorded service call, after storing filename-role tags, took 0.7202 seconds and left the source tree unchanged, with no required dataset metadata. Earlier runs during this change took 0.7571–0.9164 seconds. Timing describes the native snapshot service, excluding app process startup and UI rendering.

Classifier header discovery took 0.1789 seconds in the final run (0.1690 seconds in an earlier run). It discovered 466 diffuse originals: 321 eligible native tiers, 145 undersized originals and zero unreadable originals. It selected 148 distinct material/folder/color examples. Every selected image has both axes at least 2048; no selected file is 1K. Both Pine Bark folders are retained separately.

The selected Carpet input is `CarpetSubstance001_COMPILED_baseColor_2k.png`, a 2048 × 2048, 16-bit RGBA original. Its 1024 original remains inspectable. The 1K, 2K, 4K and 8K ambient-occlusion maps are 16-bit grayscale supplemental maps and are absent from classifier inputs.

Machine-readable evidence: [source-folder and classifier discovery](source-folder-loom-room-2026-10-12.json).

## Regression and package evidence

The final Release run executed 257 tests with zero failures and 10 opt-in skips in 46.077 seconds. It includes bare-folder membership and change observation, copied teaching tags and path normalization, manual pairing and role changes, folder-root navigation and classifier settings, real-corpus accounting and timing, native filename roles, exact crop sizing, legacy recovery, classifier eligibility and duplicate-folder identity, DTTC3 metadata, training preparation and compact-model execution. A 260 × 258 fixture produces exact 256 crops and verifies unchanged source bytes, 16-bit height codes and their float tensor values. Crop exclusions and manually selected validation splits survive preparation and cleanup without changing original-source review status.

After the startup deferral change, 99 focused Release tests passed with zero failures or skips in 4.692 seconds. These cover source browsing without checkpoint inspection, on-demand generic model reconnection, retained deferred paths and comparison selections, category-specific model loading, workshop navigation, source-folder mutation and tags. Three old inventory-summary assertions were updated to the intentional original-file and paired-set counts; their training eligibility checks remain unchanged.

Native build-tool checks and `git diff --check` passed. App-hosted tests enable testability and disable hardened runtime for XCTest injection. Production packaging uses the normal optimized Release settings and hardened runtime; the generated test plug-in is moved outside the app before packaging.

`make package` succeeded. Native bundle verification, matching version checks, deep strict signatures and installed/build recursive byte comparison passed. The production parent has `flags=0x10002(adhoc,runtime)` and no XCTest plug-in remains in the app. The closed-app installer installed `/Applications/Texture Studio.app` at 0.9.25, build 120. Its smoke check exited successfully with `Texture Studio native smoke passed`.

The actual source corpus now has 1,897 stored filename-role tags: 466 inputs, 462 heights, 462 roughness maps, 462 normals and 45 supplemental maps. Tags do not claim pixel verification before training needs it. The clear `carboard_01` typo was corrected to `Cardboard 01`. Its two legacy sample references were rebased, with their original metadata backed up in user storage. All image sizes, modification times and inodes are unchanged; ten representative image hashes and all eight Cardboard image hashes match before/after. The source root still has no `dataset.json`.

The installed native source catalog contains all 1,897 files and 462 original sets, 324 exact 2048 crop views and zero warnings. All 462 original height maps are reported as 16-bit. Both legacy Cardboard resolutions still resolve to their original maps in the corrected folder.

The installed main app was launched with the source root. Native visual automation failed with `Sky Computer Use native pipe startup failed`; no live visual click-through is claimed. UI action routing is covered by compiled regressions.

## Limits

The checks establish folder membership, UI action routing, source fidelity, crop eligibility and native training execution. They do not establish learned quality or production 2048 throughput. Learned photo preparation and final material restoration remain planned pipeline stages. The separate non-developer LoRA workflow is deferred.
