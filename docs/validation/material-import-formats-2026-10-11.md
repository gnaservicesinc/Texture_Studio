# Native material import and source format correction

## Failure and correction

The Paper003 2K surface maps used recognized ambientCG filenames. Their matching
2K color file was absent because source recovery rejected the provider's 8-bit
grayscale diffuse image. Recovery applied the RGB/RGBA normal-map requirement to
diffuse maps as well, despite native import and training already supporting
grayscale color. The resulting orphan diagnostics incorrectly suggested that the
surface filenames were unrecognized.

Both provider recovery paths now use the same role-specific storage validation:
native grayscale and gray-alpha diffuse are accepted without rewriting their
samples, normals require RGB/RGBA vector channels, and displacement retains its
16-bit precision requirement. Grayscale diffuse enters the model as three equal
RGB channels; native 8/16-bit source precision, alpha-independent color values,
dimensions, linear transfer behavior and source hashes remain covered by tests.
Originals and training crops are not resized or converted through display images.

Recognized but incomplete map sets now receive one diagnostic naming the actual
missing diffuse or surface map and its pixel dimensions. A canonical-path check
prevents duplicate warnings for `/var` and `/private/var` aliases on macOS.
Explicit `--skip-source-recovery` now applies to all native dataset operations,
including scan/import, rather than only dataset reads. Normal default setup can
still obtain genuinely missing published source tiers.

## Focused verification

| Run | Passed tests | Evidence |
| --- | ---: | --- |
| Material import, filename, PNG, dataset management and specialist regressions | 86 | `/tmp/texture-import-regression-tests.log` |
| Source recovery for both providers | 29 | `/tmp/texture-recovery-formats-tests.log` |
| Import diagnostics and explicit offline scans/imports | 6 | `/tmp/texture-import-diagnostics-tests.log` |

Each listed run completed with zero failures. The recovery tests verify native
8/16-bit grayscale and gray-alpha, exact downloaded bytes and provenance,
already-present sources without repeated requests, and rejection of scalar
normal maps. The PNG regression verifies replication into RGB tensors with the
explicit diffuse transfer contract and unchanged original codes.

A separate complete-local readiness fixture imported 14 mixed-format sets and
verified 42 native target crops against their originals with no metadata or
download transport calls. It includes ambientCG, Poly Haven and authored names,
8/16-bit gray/gray-alpha, and 16-bit RGBA. Its evidence is
`/tmp/texture-import-readiness-tests.log`.

Native build-tool checks and `git diff --check` passed.

## Real library and aborted import

The native provider path recovered `Paper003_2K-PNG_Color.png` as the exact
2048 × 2048, 8-bit grayscale original. Its 432,435 bytes have SHA-256
`f40eb96a68daef1892763a8c76565d996939d39f9659393fb4167b8257a9fbb4`.
The provider archive SHA-256 is
`b5a123a6f5de70496553ae80517812ff89b73fe003c32fdfe9898bea5d5dc23c`;
the native sidecar retains the published archive size, SHA-1, member CRC and
source identity. No original was resampled or converted to another pixel format.

The final real-corpus test then scanned, imported and rescanned the entire
`/path/to/workspace/sources_mats` library with an injected transport that rejects and
records every network attempt. All 376 original source sets across 122 families
imported successfully, with zero warnings and zero requests. The rescan correctly
reported zero new sets and 376 duplicates. Every one of the 1,543 PNGs retained
its SHA-256. Paper003's 1K height, normal and roughness training crops were compared
against native crops from all three original tiers, including the recovered 2K
grayscale diffuse. Pixel codes, precision and retained metadata matched.

After that proof passed, a guarded transaction completed the user's aborted
import into `/path/to/user/Documents/Texture Studio Datasets/Wolf`. It first
required the dataset to remain empty and native, then bound the preview and index
hashes before registering all 376 sets as original references. This import also
completed with zero warnings and zero network calls. The two opt-in real-library
and repair tests passed with zero failures.

Evidence: `/tmp/texture-import-readiness-corpus-tests.log`,
`/tmp/texture-import-readiness-corpus-proof.json` and
`/tmp/texture-import-readiness-wolf-repair.json`. Production tests are opt-in;
ordinary test runs do not modify the user's corpus or dataset.

The saved specialist settings still pointed to Chimpmonk, whose dataset folder
was no longer present. The installed app prepared a separate Wolf specialist
workspace at `~/Library/Application Support/Texture Studio/Material Workspace/Specialist Models - Wolf`
and saved Wolf as the active specialist source. Its 15 categories contain all 376
sets across 122 families, retain 24 classifier-only images and report no warnings.
The previous specialist workspace was left intact. The live receipt is
`/tmp/texture-import-specialists-wolf.json`.

## Installed build

Texture Studio 0.9.21, build 116, passed `make package`, native-bundle validation,
the model payload guard and strict signature checks. The native installer
installed the update while the app was closed. `/Applications/Texture Studio.app`
is byte for byte identical to the built bundle. Installed signature/native-bundle
checks also passed for all four material tools. The installed smoke test returned:

```text
Texture Studio native smoke passed
```

This evidence verifies the real import and crop workflow. Interactive GUI
verification was unavailable because the native computer-use bridge returned
`Sky Computer Use native pipe startup failed`; no click-through is claimed.
