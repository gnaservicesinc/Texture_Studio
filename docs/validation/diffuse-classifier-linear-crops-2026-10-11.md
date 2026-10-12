# Classifier dataset, exact crops and linear input validation

Texture Studio 0.9.19, build 114.

## Created classifier dataset

`/opt/ipde/classifer_ds` contains 120 original PNG diffuse examples and three added
Other Diffuse examples, in fifteen class folders. Other Fallback remains empty
for user-supplied examples. The review
page is `/opt/ipde/classifer_ds-review.html`; the source manifest is
`/opt/ipde/classifer_ds/.classifier-manifest.json`.

Actual Create ML ingestion rejected the two empty Other folders. They remain
available for user examples in the review dataset. Refresh creates
`/opt/ipde/classifer_ds-createml` with only populated class folders for Training
Data; this copy also preserves image bytes and dimensions.

| Class | Images |
| --- | ---: |
| Bricks | 11 |
| Cement Concrete Pavement | 6 |
| Wood | 22 |
| Bark | 4 |
| Small Particles | 14 |
| Ground | 11 |
| Medium Particles | 6 |
| Metal | 8 |
| Rock Cliff Marble | 7 |
| Tiles and Pavers | 15 |
| Fabric | 9 |
| Paint Wallpaper Drywall Stucco | 6 |
| Plastic Rubber Clay | 1 |
| Other Diffuse Textures | 3 |
| Other Fallback | 0 |

All 120 original files and classifier copies were independently checked against
the recorded SHA-256 values, and image dimensions were checked with ImageIO.
`/opt/ipde/classifer_ds-verification.json` records that verification. A second
native CLI refresh added zero files and preserved all 120 images. Initial
labels use Poly Haven provider tags, material names and visually reviewed
mixed-material examples; all suggestions remain marked for user review.

Bark was split from Wood at the user's request. The native refresh reported four
reclassified images, zero added images and 120 preserved examples. All 120
source/copy hashes and dimensions passed verification again after the split.
Descriptive material-family
filenames are retained, and conservative subtype suggestions are recorded in
the manifest and review page for later specialization. Taxonomy migration
preserves manually moved, renamed, edited and removed examples.

The final native refresh preserved all 120 source-linked examples and the
additional Other Diffuse images. The actual Create ML
`labeledDirectories(...).labeledImages()` API loaded 123 images across 14
populated labels. Every training-view copy matched its curated image's SHA-256
and dimensions. `/opt/ipde/classifer_ds-createml-verification.json` records this
import and byte verification.

The final installed native CLI refresh reported 123 images, 14 populated
training labels, `training_ready: true`, zero additions to the source-linked
inventory, 120 preserved source examples and no warnings.

## Native regression checks

The twelve final classifier service tests passed, including incremental refresh,
preservation of manually moved/renamed/edited/deleted images, user examples,
uncertain materials, original-source discovery, destination checks, Bark
taxonomy migration, subtype hints and occupied migration destinations. They
also cover byte-preserving publication of a populated-class Create ML view,
ownership checks and actual Create ML ingestion of 120 fixture images across
13 populated labels, with both Other review folders empty.

The broad native run executed 557 tests with 16 optional skips and one failure.
It passed all 48 dataset and 12 crop augmentation tests,
including exact 8/16-bit code preservation, normal axis transformations and
rejection/rebuilding of declared resampled crop caches. It found a new compact
baseline comparison that omitted the checkpoint's input transfer during
inspection. The inspection now carries that contract to comparisons.

After that correction and the final UI changes, the focused native run passed
80 tests, with two optional fixture skips and zero failures. It covered compact
model training/inference/package compatibility, linear tensor transfer,
safetensors, classifier UI routing, preferences and Trainer layout. Its result:
`build/TextureStudioClassifier/Logs/Test/Test-TextureStudio-2026.10.11_19-35-27--0400.xcresult`.

Final integration review also corrected fresh-baseline convolution padding:
comparison models now inherit the checkpoint's padding as well as its input
transfer. All ten compact model tests passed after that correction. The two
classifier UI tests passed with taxonomy-change counts visible in the result.
Final focused evidence:

- Classifier service and final UI: `build/TextureStudioClassifier/Logs/Test/Test-TextureStudio-2026.10.11_19-59-02--0400.xcresult` (14 tests, zero failures).
- Compact model: `build/TextureStudio/Logs/Test/Test-TextureStudio-2026.10.11_19-46-35--0400.xcresult`.
- Classifier UI: `build/TextureStudioClassifier/Logs/Test/Test-TextureStudio-2026.10.11_19-47-30--0400.xcresult`.

## Package and installation

`make DERIVED_DATA=/opt/ipde/ipde/build/TextureStudio package` passed after the
populated-class import view was added.
The build-tool and native installer regression checks passed. The packaged
parent and four embedded material applications passed native bundle validation
and strict code-signature verification.

Installation with the native Swift installer and `--if-closed` succeeded.
`diff -qr` found no differences between the built and installed application.
The installed application reports version 0.9.19 and build 114; strict signature
and native bundle verification passed. Installed smoke output:

```
Texture Studio native smoke passed
```

## Scope

The native computer-use pipe was unavailable, so no live Create ML import or
visual click-through is claimed. Choose `/opt/ipde/classifer_ds-createml` as Training Data
in the supplied Create ML project after reviewing labels. No classifier was
trained and no automatic specialist routing was enabled.

The supplied displacement files confirm real contrast collapse, quantified in
[the linear-data report](../material-linear-data.md). These changes do not
establish improved learned quality. New compact RGB runs use linear-light
diffuse input; existing checkpoints preserve their original input convention.
Numeric material targets and Float32 EXR output preserve their data values.
