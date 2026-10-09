# Native material tools

Texture Studio's **Model Training** workspace prepares paired material maps, trains a local material LoRA, reviews results, and exports or uploads the selected model. Material Review, Checkpoint Compare, Material Dataset, and Material Trainer also launch independently from the main app bundle.

```sh
./script/build_and_run.sh run --tool review
./script/build_and_run.sh run --tool compare
./script/build_and_run.sh run --tool dataset
./script/build_and_run.sh run --tool train
```

Builds use Release unless `--debug` is selected. `make package` includes the tool suite. Datasets, model weights, generated maps and app bundles stay outside Git.

## Create and manage datasets

Open **Material Dataset** from Model Training. **New Dataset…** creates a named dataset with a rendering/training resolution chosen before import; choose a save location or use the displayed Documents default. **Open Folder…** opens a dataset folder directly or routes a raw folder into dataset setup/import. Dropping a folder works the same way. Recently opened datasets appear under **Switch Dataset**. Use **Dataset Info…** to rename the dataset, edit its description, or change resolution while reviewing the resulting crop and split counts. The resolution is stored with the dataset and restored when switching datasets.

**Add Materials…** registers a named material with a diffuse PNG and at least one displacement, roughness or normal PNG. Choose matching source dimensions and the correct normal convention. **Import Folder…** recursively discovers paired maps and previews the resulting dataset at every supported grid before import. It reports duplicates, unsupported files and invalid maps; valid sets are only added when Import is pressed. Progress shows the folder being inspected. An unchanged scan is reused across resolution changes and registration; changed originals or dataset reviews require a rescan. Both actions reference the original full-quality files without copying or changing their pixels. An empty dataset can be created before any materials are added.

Select a material to edit its review note, approval status or **Split**. The list shows training and validation materials; the **Show** filter can narrow it. Manual split assignments are retained when training maps are prepared. The source-family guard rejects assignments that would put related resolution sets or diffuse variants in both training and validation.

**Remove Material…** removes that material's dataset entries while keeping its original map files. **Delete Dataset…** asks for confirmation, moves the dataset metadata to Trash and removes it from the recent library. Original source maps remain in place. Restore the metadata from Trash to reopen a deleted dataset. Creation and edits keep their form values if a save fails, so you can correct the problem and retry.

## Dataset pixels and storage

The original source folders are authoritative; importing `/opt/ipde/sources_mats` includes its nested material folders. Each dataset keeps an index of registered resolution sets and diffuse color variants. `samples/<material>_full/sample.json` contains small manifests, not copies of the source images.

Select a **Training map size** supported by the current model, available memory, and original source dimensions. The memory budget follows resolution automatically: final-map LoRA uses a 24 GiB recommendation at 1K and 48 GiB at 2K with the default 0.5 GiB map cache, when the hardware has room. Developer mode exposes a manual override. The 2K displacement path completed a real float32 forward/backward/optimizer probe at about 42.3 GiB peak Metal memory; this does not establish model quality or extended-run stability. Every diffuse and numeric target presented to training has exactly that square grid. The model receives the entire displayed native crop, with no smaller random crop, exposure augmentation, or hidden resize. A 2K run ignores all 1K sets. Training availability and generation/export resolution are separate controls.

At the selected dimensions, unchanged maps use their original paths. Larger 4K sources produce one center crop. Sources at least 8K on both axes produce three disjoint corner crops, with a held-out corner when regional validation is eligible. Every map and diffuse color variant uses the same source rectangle. Each changed crop is saved once under `.training-data/`; numeric integer codes retain original precision with no resampling, gamma, range stretching or denoising. The diffuse color picker previews each registered variant, and training randomly selects among them for the same target geometry.

Only one temporary training view is retained. Switching size removes the previous view after saving its review fields. Completion or cancellation removes prepared images and manifests, returns the workspace to the original source dataset, and keeps small review records per size. Review recreates the exact selected grid from the original in memory when needed. This avoids permanently storing several resolutions or full source copies.

Automatic validation keeps all resolutions and colors of a source family together. Regional checks are limited to disjoint regions of a single included resolution set and are labeled as checks of a known material. A one-region fit has no independent validation set. Approval, exclusion and notes update metadata atomically; they never rewrite source images. Approval at one resolution does not automatically approve another.

Opening a dataset and choosing its display grid allocate no training images. Inspection reconstructs only the viewed map into a temporary file and removes it after decoding. Size-specific review decisions live in `.material-size-reviews.json`; the exact native crops are staged when training starts and purged afterward.

**Stop** aborts preparation or model setup immediately. During training, it aborts without requesting a new checkpoint; files already saved on disk remain available. **Stop and Save** appears once training updates begin. It finishes the current update and saves the model using the selected export mode, then defers new comparison renders to a later review. Stop remains available while saving, so a pending save can also be aborted. Neither action changes original source maps.

If an unresponsive worker has to be forcibly terminated while creating crops, its incomplete temporary stage is removed during the next preparation.

## Review and inference

Map inspection retains the original file separately from its display representation. **Export Original** preserves original bytes. Display contrast and training-grid views do not alter numerical exports. Missing generated maps can be reconstructed from intact sources. A listed sample is removed only after its original source is genuinely missing and cannot be recovered at another recorded location.

Comparison labels identify the source, target, model/checkpoint and training step. Predictions use unclipped Float32 EXR; display PNGs are separate views. Review can ask the local decision model to assess detail, visual appeal and artifacts and preselect a recommendation. The user controls the saved decision.

A test photo goes through Texture Studio's geometry, illumination and color preparation before inference. The model receives the same diffuse map used by material generation. Passing an untreated photo to the backend is rejected. No spatial denoiser is applied to the photo; registered companion evidence can reduce capture noise while retaining detail.

## Model exports and Hugging Face

Normal mode saves a `.safetensors` LoRA. Enable **Developer mode** in Settings to expose rank, alpha, refinement scope, base override and upload controls, and to export a full fused `.safetensors` checkpoint plus the separate LoRA. The full checkpoint contains the base with adapter deltas merged; both outputs retain the exact base identity and model metadata. Export packages exclude source photographs and optimizer state.

Compatible LoRAs can be combined with explicit weights. Compatibility requires the same base identity, target and module layout; image size alone is insufficient. The individual LoRA remains available alongside a fused export.

Use **Refresh Account** to read a saved Hugging Face login. Upload uses the selected model, destination and visibility shown in the workspace. Developer mode offers upload after training. The saved Hub catalog includes successfully uploaded models with a **Download** action, so removable local files can be obtained again from their recorded repository revision. Base removal affects only app-owned files; external bases are unlinked.

The current material backend is an experimental native-scale PBRnxt adaptation. A future V1 catalog is intended to contain refined full checkpoints for height, roughness and normals. The shipped quality must be established through full-resolution map and displaced-surface review; successful training or a lower loss does not establish that quality.
