# Diffuse texture classifier dataset

Dataset Studio has a separate **Create / Refresh Classifier Dataset** button.
Its settings select the original material source folder and classifier output
folder independently of the material training dataset. Leaving the source
folder empty uses the open material dataset's original source inventory.
Refreshing material crops does not refresh this classifier dataset.

The initial dataset for `Classifiers/diffuse_texture_type/Diffuse Texture Type
Classifier.mlproj` is `/opt/ipde/classifer_ds`. Review and edit labels there.
Choose `/opt/ipde/classifer_ds-createml` as Training Data in Create ML. It is a
generated copy containing only populated classes. The review dataset contains
fifteen current material class folders, with 148 diffuse images after the user's
classifier training and added Paper/cardboard materials. Refresh preserved all
147 existing images at their exact paths and hashes and added Cardboard002. The current taxonomy includes Paper,
Drywall and Leather. Existing datasets use their current immediate class folders;
refresh does not recreate deleted old category folders. Empty folders are omitted
from the generated Create ML training view.

Bark has its own class, separate from processed Wood. The initial split has four
bark images and 22 wood images. Names remain descriptive and unique, such as
`hinoki_planks--color_1--379cffbb30e5.png` and
`oriented_strand_board--color_1--c34c73deb4c5.png`. Create ML's folder-based input
uses the parent folder as the class label; words in the filename do not teach
additional subtype labels. Keep the material family and color variant in the
name rather than reducing every filename to a generic type.

The manifest and review page also record a conservative `subtype` suggestion,
such as planks, boards, beams or panels when explicit name/tag evidence exists.
This supports later subtype splits while preserving each source's identity.
Uncertain subtypes remain unset. Keep images directly inside the current class
folder; build a separate subtype dataset when training a finer classifier.

One smallest original tier is copied per material family and color variant.
Repeated 1K/2K/4K tiers of the same material are omitted to reduce validation
leakage. Files keep their original dimensions, pixel codes, color metadata and
SHA-256 identity. No classifier dataset image is resized, cropped or augmented.
Create ML's image feature extractor can resize its own input internally; this
does not alter these files or the depth/material training crop contract.

Labels use material names, filenames, live Poly Haven tags, and recorded visual
review of mixed or misleadingly named surfaces. All initial suggestions remain
marked for user review. `/opt/ipde/classifer_ds-review.html` provides thumbnails,
suggested classes, evidence, original dimensions and source hashes. Correct
labels by moving images between class folders in Finder. Refresh preserves
manual moves, renames, edits, removals and added images. A taxonomy update, such
as splitting Bark from Wood, updates unchanged automatic suggestions; it leaves
manually curated files in place. Unrecognized materials
are kept outside the Create ML parent in a sibling pending-review folder.

Refresh also rebuilds the sibling `-createml` training folder from populated
class folders. Correct labels in the review dataset, then refresh. The generated
training copy preserves every image's bytes and includes the Other classes
once you add examples. Its ownership manifest protects unrelated files and
user edits from replacement; a view-publication problem is reported as a
warning while the completed review dataset remains available.

The current Create ML API check loaded all 148 images across 15 populated classes,
with matching original bytes and dimensions.

The hidden `.classifier-manifest.json` records each original image and current
label. Refresh is locked, checks copied bytes before publication and saves
progress incrementally. A provider outage falls back to name-based suggestions
and records a warning. It does not prevent review of the available images.

For headless refresh using the native application:

```sh
'/Applications/Texture Studio.app/Contents/MacOS/Texture Studio' \
  --refresh-classifier-dataset \
  --source-folder /opt/ipde/sources_mats \
  --destination /opt/ipde/classifer_ds
```

The exported IFPv2 classifier is integrated into the specialist workflow described
in [Specialist Models](material-specialists.md). Folder labels remain authoritative
when a curated diffuse image matches a paired material source. Classifier-only
images remain preserved examples and do not become fabricated depth targets.

Review the labels before classifier training. For an independent accuracy test,
keep complete material families out of training, including their color
variants. Known-corpus runtime checks do not establish accuracy on novel materials.

Folder-label behavior follows [Apple's Create ML labeled-directory
documentation](https://developer.apple.com/documentation/createml/mlimageclassifier/datasource/labeleddirectories(at:)).
