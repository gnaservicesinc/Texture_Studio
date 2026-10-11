# Rotation, dataset cache ownership, and scalar EXR compatibility

Version 0.9.18, build 113. Evidence is retained outside the repository at
`~/Library/Application Support/Texture Studio/Validation/rotation-cache-cleanup-2026-10-11`.

## Reported failures and changes

Arbitrary-angle center spins rounded source coordinates and copied the nearest
integer pixel. This preserved individual codes but introduced artificial stair
steps into continuous height targets. Center spins now interpolate at pixel
centers, round once to the original 8/16-bit storage, and record
`bilinear-pixel-center-v1`. Matching diffuse/target geometry, exact aligned
quarter turns, original-source reads, and no-padding crop support are retained.
Normal vectors are interpolated, rotated, and renormalized before encoding.

Training predictions now live in the original dataset's
`.training-data/depthpro-inputs`, and mutable pairs/history in isolated
`.training-data/run-crops/<run identity>` directories. Each run's
`crop-cache.json` points to its crop directory. Explicit cache overrides remain
supported. Entries from the former shared cache are validated and adopted
lazily using hard links when available; cross-volume reuse does not duplicate
the whole cache. Inference retains its shared default.

The failed prepared directory contained only recognized Finder `.DS_Store`,
with normal ownership, permissions, and no ACL or immutable flags. The failure
occurred after its generated contents and manifest were removed. Cleanup now
removes only verified owned files/directories, retries late Finder metadata
creation with a bounded `rmdir` loop, and keeps a recovery receipt when final
removal fails. Reusable prediction/run caches survive prepared-size cleanup.
Source reopening is attempted independently, including after post-save
cancellation, so cleanup cannot strand the UI on a partially removed dataset.

Scalar Float32 EXRs previously declared the sole channel `R`. GIMP correctly
interpreted this as an RGB red channel. Both native Float32 writers now declare
`Y` luminance; RGB/vector exports remain `B`, `G`, `R`. Installed GIMP 3.2.6
imported the old displacement as RGB and the corrected copy plus both writer
fixtures as grayscale at the same Float32 linear precision. The corrected
existing EXR changes only the channel-name byte; the numeric payload is exact.

## Saved model and boundary evidence

The stopped model at step 1,534, four complete epochs, remains unchanged:
`85f28673fe3cf8a1cdd6adbbb3695ca451dc12dbbd1bb745771d7217f8cd9c29`.
Validation was disabled. Its metal-texture displacement has faint relief and
a prominent perimeter frame; it is not numerically constant. Native Float32
output has standard deviation 0.004189, with 98% of values between
0.476746 and 0.483499. Its source reference height has standard deviation
0.122115. This observation does not establish rotation as the cause of weak
diamond relief.

The application EXR's outer-ring standard deviation is 0.060628 versus
0.000894 in the interior. A separate comparison using the same checkpoint
shows oscillating output-edge bands while its prepared DepthPro input edge
varies smoothly. The frame exists before EXR encoding or normal conversion.
The compact graph used zero padding at its spatial convolutions.

New compact scalar, normal, and DepthPro-refiner models record
`convolution_padding=reflect-v1`. Spatial convolutions reflect their feature
borders and retain the exact model input/output grid. Legacy checkpoints without
this field use `zero-v1`, preserving their original numerical behavior. Unknown
policies are rejected. A constant-field regression compares both policies;
reflection gradients are checked against finite differences. This boundary fix
does not claim that the saved checkpoint has learned the missing diamond relief.

## Verification

Validation covers analytic scalar ramps, sharp-edge coverage without overshoot,
paired coordinates, normal-vector length, exact quarter turns, unchanged
originals, cache source/model identity and corruption, legacy-cache adoption,
late Finder metadata, partial-cleanup recovery, post-save source reopening,
and bitwise Float32 EXR roundtrips. Optional retained cedar exports provide an
actual corrected 1024 crop at 174 degrees.

The focused Release run selected 148 tests. Cleanup receipt enumeration initially
failed because Foundation returned `/private/var` while the canonical dataset
used `/var`; enumeration now canonicalizes and checks containment before deriving
relative paths. A cache-location assertion also compared URL directory flags
instead of paths. The final rerun passed all 47 dataset tests and the corrected
trainer assertion (48 tests, zero failures). All other selected tests passed in
the original run. Three opt-in production/benchmark checks were skipped; this
was not a complete repository test run.

The real installed Core ML DepthPro model completed one native 256-square
training update, exported a reflection-policy checkpoint, and passed inference
and Float32 EXR readback. The nine compact-model tests include reflection's
constant-field boundary behavior, finite-difference gradient agreement, and
bit-exact legacy checkpoint loading. No training run was restarted on PolyMat.

The failed Finder-only directory was archived under the validation directory;
the original PolyMat index remained
`6aedc086e6b41c53d990c84423034ab9d8ff6ac67710b31e4d87c7a2a3404bae`.
The user export also has an adjacent `displacement-grayscale.exr` copy for
inspection in GIMP. Its correction preserves the old checkpoint's faint relief
and edge bands; it does not regenerate or improve the prediction.

`make package` completed with normal Hardened Runtime enabled and without the
XCTest plug-in in the shipped bundle. Native bundle checks, deep strict code
signature verification, version agreement, and the Release Material Trainer
launch passed. The native installer replaced the closed installed app with
0.9.18 (113). The installed bundle matched the built bundle byte for byte, and
all five packaged executable hashes matched their installed copies. Installed
smoke output: `Texture Studio native smoke passed`. The installed Trainer was
opened on the original PolyMat dataset without starting training.

Model quality remains an evaluation question. Successful updates, numeric
exports, and regression checks do not qualify restored diamond detail or
production material quality.
