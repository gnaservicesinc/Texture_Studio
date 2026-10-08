# Native material tools

Texture Studio includes a **Model Training** button in its toolbar and sidebar. It opens one guided workspace for preparing data, training/refining, comparing results and exporting a model. The four tools are also independently launchable inside `Texture Studio.app/Contents/Applications/`, sharing the same Swift UI components and bundled local Python backend:

| App | Main workflow |
| --- | --- |
| Material Review | Inspect original PNG/EXR maps, synchronized navigation, full-quality pop-out windows, and export originals for GIMP. Open the packed Blender review scene to inspect actual displaced geometry. Save human review decisions separately. |
| Checkpoint Compare | Select two or more compatible checkpoints, run the same photo through each on Metal, and inspect the resulting native maps together. Each run records the actual checkpoint hash. |
| Material Dataset | Select a material crop, inspect diffuse/height/roughness/normal originals, approve/exclude/reapprove, edit review notes, and assign training/validation metadata. |
| Material Trainer | Choose a dataset, native 1024/2048 size, height/roughness/OpenGL-normal target, warm start, updates per crop and resource limits. Run, resume, stop and save; inspect live logs and run folders. Manage, package and optionally upload checkpoints. |

Build and open any tool from the repository:

```sh
./script/build_and_run.sh run --tool review
./script/build_and_run.sh run --tool compare
./script/build_and_run.sh run --tool dataset
./script/build_and_run.sh run --tool train
```

Normal builds and Run actions use **Release**, under `build/TextureStudio/Build/Products/Release/Texture Studio.app`. The embedded tools are proper `.app` bundles with separate names and bundle identifiers, compiled through the existing Xcode project. The tool role is a bundle setting; `--tool ROLE` can also select a role for development. `--debug` explicitly chooses Debug and launches LLDB; working with your materials does not require a Debug build. `make package` includes all four tools, and `make install` installs the whole suite in `/Applications/Texture Studio.app` after checking that the installed app and its tools are closed.

Review can open a `review-manifest.json`, a saved review, or original image maps directly. `--review /absolute/path/review-manifest.json` selects a review at launch; `--dataset /absolute/path/dataset.json` selects a dataset. The current local 2K review opens on first launch when available. Maps retain their original dimensions; numeric-map display conversion affects only the screen image. Export Original copies the source file byte for byte, including its original PNG/EXR precision. GIMP receives a byte-identical editable copy, preserving dataset originals. Saved reviews reopen their per-candidate decisions and notes. Review decisions never select a production model automatically.

Use Runtime to locate Python, a working folder, and the pinned DINOv2 weights/source. A new installation creates its default working folder under Application Support when it first produces results. Existing runtime choices are shared across the tools and persist after installing a Release build. Python needs the project's dependencies plus the training dependencies (`Pillow`, `safetensors`, and the pinned encoder source); Hugging Face upload additionally needs `huggingface_hub`. Missing files are reported so they can be reconnected. Download Pinned Encoder installs an app-owned, checksum-verified copy; Remove Downloaded Encoder removes only that managed copy. Located external files remain in place.

Use Checkpoints → Locate to inspect the checkpoint schema, target, step and SHA256. Checkpoint filenames do not select architectures. Incompatible models fail explicitly. A comparison uses the same native source dimensions for each candidate, and a new output folder; it records a reopenable review manifest. The model's small contextual encoder grid does not replace the native material-head output grid.

Dataset edits change metadata only, with an expected index hash and recovery journal. The backend refuses stale selections and active training edits. Original PNG data, encoded values, precision and color tags are not rewritten. Region validation still measures learning on familiar source materials; fresh materials are needed to assess generalization.

Changing native crop size automatically prepares matching crops from the recorded original maps, then selects that size-specific dataset. It does not scale a 1K crop to 2K. Generated variants are cached; the original dataset and review history are retained. Each size has its own split lineage. Validation on a new size after fitting another size is familiar-source feedback, not an independent generalization test. Keep the originals available when asking for a larger crop; after preparation, training can use the self-contained crops.

Stop & Save sends SIGINT to the trainer, which retains completed optimizer updates. Resume uses `checkpoint.latest.pt` with optimizer state, rather than treating a weights-only checkpoint as resumable. Set the desired total updates per crop when extending a run. Choose **Start from base** for a new head using DINOv2 Base features, or **Refine checkpoint** to start a new run from learned weights. Refining an existing rank-8 DINOv2 LoRA checkpoint keeps its encoder adapters and contextual resolution fixed while training the material head; saved checkpoints and exports retain those adapters. Full dataset fitting, roughness and direct normal fitting are explicit training runs; opening the UI never starts one.

Export Package writes a compact inference checkpoint, model card, source/license information, checksum manifest and pinned encoder dependencies into a new directory. It does not embed the pretrained encoder, training source images or optimizer state. A frozen encoder plus learned material head is identified as that architecture, not as a DA3 LoRA. Upload Exported Package requires an explicit `owner/model-name` and a saved Hugging Face login (`hf auth login`); new repositories default to private. No token is displayed or stored by the Swift app. Upload is never automatic.

Use in Texture Studio records the selected height checkpoint hash and runtime dependencies. Texture Studio's Material checkpoint source loads that exact checkpoint on Metal. It preserves the learned height amplitude instead of applying camera-depth percentile normalization. Perspective/crop and explicit relief contrast still apply. This native material head currently supports 1024/2048 output; larger output is refused with an explanation. Roughness and direct-normal checkpoints can be trained and compared; Texture Studio currently integrates height checkpoints and derives OpenGL normals from that height.

Development commits go to `main`. Datasets, model weights, app bundles and generated review outputs stay outside Git. Enable Git LFS before publishing embedded model weights. Local experiment history and recovery records are preserved separately.
