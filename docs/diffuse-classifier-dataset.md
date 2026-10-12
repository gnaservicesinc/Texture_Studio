# Diffuse texture classifier dataset

Start in **Loom Room** with a bare source folder, such as `/opt/ipde/sources_mats`. **Manage Images & Tags** lets you label each diffuse image inside the app. Use **Prepare Classifier Examples** when you are ready to make the review and Create ML datasets. Browsing originals does not run classification, load a model or hash every image. See the [source-folder workflow](source-folder-workflow.md).

**Classifier Settings…** uses the open source folder by default. Enable **Use another original source folder** to choose an independent source explicitly. The classifier output folder remains a separate choice. Refreshing material crops does not refresh classifier examples.

## Which originals become examples

New source folders and independent classifier sources default to a 2048 grid. An original qualifies only when its actual width and height both supply the selected grid. A filename containing `2K` cannot make a 1024 image eligible.

For each material family and diffuse color variant within a real source folder, refresh selects the smallest native original that supplies the grid. For example, 2048 training selects a usable 2K tier instead of its 1K copy, and can use a larger tier when 2K is absent. Same-named families in different original folders remain separate. Repeated tiers from one family and variant are omitted from the Create ML view to reduce validation leakage.

Smaller originals, previously selected tiers and user-curated examples remain available for review. Examples below the selected grid are omitted from the generated training view. Files retain their original bytes, dimensions, pixel codes and color metadata; classifier dataset preparation does not resize, crop or augment them. Create ML's feature extractor may resize inputs internally without changing these files or the material crop contract.

Only recognized diffuse filenames or images explicitly tagged as diffuse supply examples. AO, packed masks and surface targets have separate roles. Unassigned images stay visible in the original source folder so you can teach their role and material type.

## Review, train and export

Set the material type in **Edit Tags…**, using a listed class or a custom label. Valid user file tags take priority over automatic suggestions. Local names, saved provider metadata and prior curation supply suggestions for the remaining examples. Uncertain labels remain marked for review; refresh does not wait for an online provider catalog.

The default review folder is `/opt/ipde/classifer_ds`, and `/opt/ipde/classifer_ds-review.html` shows thumbnails, suggested classes, evidence, original dimensions and source hashes. The review folder's immediate class folders determine Create ML labels. Correct original tags inside the app and refresh; existing manually curated class-folder moves, renames, edits and removals remain supported. Edit the review dataset rather than its generated `-createml` view.

Choose `/opt/ipde/classifer_ds-createml` as **Training Data** in Create ML. It contains populated classes and currently eligible examples only. Train the image classifier, export its Core ML model, then choose **Choose Trained Classifier…** in Loom Room and refresh the material categories. Classifier-only examples remain preserved; height, roughness and normal training each require suitable paired targets.

Keep descriptive family and color-variant names. Create ML learns the parent class folder; filename words do not teach extra subtype labels. The manifest and review page may suggest subtypes such as planks or panels, but finer classification requires its own labeled examples and retraining. For an independent accuracy test, keep entire material families and all their color variants outside training.

## Generated files and compatibility

The original source root requires no dataset JSON. The generated classifier review dataset has an app-owned `.classifier-manifest.json` for provenance and preserving curation. Refresh checks copied bytes, saves progress and rebuilds the sibling `-createml` view. Unrelated files and user edits are protected from replacement; a view-publication problem becomes a warning while the review dataset remains available. Unrecognized examples stay in a sibling pending-review folder.

Legacy indexed material datasets and existing class-folder review datasets remain readable. An earlier classifier snapshot contained 148 images across 15 populated classes; those historical counts do not describe current folder membership. Refresh uses the current source grid and eligible native tier selection, so older 1K review copies can remain without entering a new 2048 Create ML view.

For an explicit native command-line refresh:

```sh
'/Applications/Texture Studio.app/Contents/MacOS/Texture Studio' \
  --refresh-classifier-dataset \
  --source-folder /opt/ipde/sources_mats \
  --destination /opt/ipde/classifer_ds \
  --size 2048
```

Folder-label behavior follows [Apple's Create ML labeled-directory documentation](https://developer.apple.com/documentation/createml/mlimageclassifier/datasource/labeleddirectories(at:)). Known-corpus checks do not establish accuracy on novel materials.
