# Exact material training crops

The crop audit found no spatial resize in native dataset preparation. The
selected diffuse and target maps use `NativePNG.crops` with integer source
rectangles, and original-size maps stay as checksum-bound source references.
Mismatched paired native dimensions and undersized fields are rejected; the
preparer does not scale a smaller field to satisfy the training size.

The center-spin augmentation did resample diffuse/target pixels with bilinear
interpolation at arbitrary angles. That preserved spatial scale, but it created
new pixel codes. New center rotations take an exact source crop and use only
90, 180 or 270 degree pixel permutations. Flips retain their existing optional
setting. Scalar/diffuse codes remain exact; normal X/Y channels are swapped
and/or complemented to follow image axes, with Z unchanged. There is no vector
renormalization or interpolation. Odd-sized original fields use a single
integer-aligned center rectangle, and native-sized center fields need no border
support or padding. New receipts/configuration identify
`exact-quarter-turn-pixel-permutation-v1` and record that source pixels were
neither resized nor interpolated.

Prepared-cache reuse now validates its declared pixel transformations in
addition to dimensions and hashes. Prepared maps require an exact native-crop
or original-source-reference binding. Declared resize/resample transformations,
gamma/range alterations in crops, non-exact crop flags, and unknown transforms
are rejected. Training applies the same check to the selected target and every
diffuse variant. An invalid prepared cache is rebuilt from its unchanged
original dataset through the existing owned-cache preparation path.

DepthPro's required 1536-square model input is separate from the native
diffuse/target crop grid. Its model-input resize remains recorded by the
DepthPro preparation contract; it is not performed by native crop preparation.
The legacy metadata key `source_resize_algorithm` also appears in comparison
receipts, but its `exact_native_crop` value describes an exact crop rather than
a resize.

## Retained cache audit

The current Chimpmonk dataset has no prepared-size directories in
`.training-data`; its remaining directories are `depthpro-inputs` and
`run-crops`. Its saved run histories were read without modification:

| Run identity | Entries | Exact crop/flip | Bilinear center spin |
| --- | ---: | ---: | ---: |
| `bd322b7bf4c7f111ccf73c35` | 1,116 | 722 | 394 |
| `1b4ec836498fdf2fcd922522` | 7,380 | 7,380 | 0 |
| `601e4aa0f699205de97bf6d3` | 1 | 1 | 0 |

The preserved history SHA-256 values are, respectively:

```
946e6470f967d3e17ed7fee6af114b5464eccb76b270e8795610e91b77834127
d1abd63a09d5039f8b8a3e8b1d4dbc8d8888832c5db4bf513e992820461cc7ce
883abdec52209df3fcdf158b3e9724d78b8ab4a82d6858bcd662b279966a62fc
```

The original Chimpmonk index hash at audit time was
`419b6497736985d6868a1622c354bd76673224c9bb2f61ca6c95e11d162282cc`.
Older run-owned interpolated crops remain saved provenance. Each new training
output owns a new run-cache directory, so it does not adopt those mutable
interpolated crops. This change cannot improve an already-trained checkpoint.

## Regression coverage

`NativeMaterialCropAugmentationTests` checks exact 8/16-bit paired code
membership on odd source grids; quarter-turn inverse round trips; sharp edges
without intermediate codes; rejection of interpolated angles; native-sized
support without padding; reversible normal-axis transformations; per-epoch
paired geometry; unchanged compressed bytes when augmentation is disabled; and
reuse/remapping of DepthPro predictions for exact orientation changes.

`NativeMaterialDatasetTests` checks exact generated crop codes and unchanged
originals. Its new resampled-cache regression retains valid dimensions and file
hashes while injecting resize transforms, gamma alteration, non-exact flags,
and a scaled diffuse variant. Both cache reuse and training reject these
metadata contracts, and automatic preparation rebuilds exact crops while
preserving the original source bytes.

The consolidated native run passed all 12 crop augmentation tests and all 48
dataset tests. Package and installed verification are recorded in the
[classifier and linear-input validation report](diffuse-classifier-linear-crops-2026-10-11.md).
