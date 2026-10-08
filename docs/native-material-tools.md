# Native material tools

Texture Studio’s **Model Training** button opens dataset preparation, full-resolution review, legacy checkpoint comparison and export. DINOv2 material training is retired: new runs, resume and **Use in Texture Studio** are unavailable for those checkpoints. Retirement itself does not delete original datasets or saved checkpoint files. Comparison/export requires those files and their recorded dependencies; externally removed files must be located again. A replacement texture-height backend must pass the [model acceptance checks](material-model-vetting.md) before training and Studio activation return.

The four tools also launch independently from `Texture Studio.app/Contents/Applications/`, sharing Swift UI components and a bundled local Python backend:

| App | Current workflow |
| --- | --- |
| Material Review | Inspect original PNG/EXR maps with synchronized navigation and full-quality pop-outs. Export originals or open copies in GIMP. Open the Blender review scene to inspect displaced geometry and save review decisions separately. |
| Checkpoint Compare | Compare saved experimental checkpoints on the same source; each result records its checkpoint hash. A comparison is a research review, not production activation. |
| Material Dataset | Select paired crops, inspect diffuse/height/roughness/normal originals, approve or exclude crops, and save notes. Automatic check crops stay out of normal browsing. |
| Material Trainer | Prepare native 1024/2048 paired crops and inspect saved operation logs or checkpoints. New training and resume are paused pending a validated replacement backend. |

```sh
./script/build_and_run.sh run --tool review
./script/build_and_run.sh run --tool compare
./script/build_and_run.sh run --tool dataset
./script/build_and_run.sh run --tool train
```

Normal builds and Run actions use **Release**, at `build/TextureStudio/Build/Products/Release/Texture Studio.app`. Embedded tools have separate names, bundle identifiers and icons. `--debug` explicitly chooses Debug and LLDB. `make package` includes the suite; `make install` installs it into `/Applications/Texture Studio.app` after checking that Studio and its tools are closed.

## Native dataset size

Changing **Native crop size** automatically prepares matching crops from recorded original maps and selects the size-specific dataset. Opening a dataset or restoring saved settings also checks the remembered size against every map’s actual PNG header. Missing or mixed-size maps can be rebuilt from their original parents; metadata declaring “2048” alone is insufficient. A 1K crop is never upscaled into a 2K training target. Original maps, previous datasets, review notes and exclusions are retained.

On the current development Mac, **`/opt/ipde/material-dataset-2048`** is the visible alias for the active 2K dataset. Its 199 crops contain 796 paired maps, all independently verified as **2048 × 2048** with matching SHA256 hashes. `/opt/ipde/material-dataset/samples/` contains historical 1K crops, including `winter_leaves_auto_003`; those are not the active 2K output. The original source maps remain under `/opt/ipde/material-dataset/sources/`. The Dataset view shows the selected folder and verified actual dimensions. Reveal that folder to inspect the active files.

Preparation runs materials concurrently using available CPU cores within memory headroom, with light lossless PNG compression (level 3). Exact existing crops are checksum-verified and copied without recompression; APFS can use independent copy-on-write clones. Each original map is decoded once per material/role for missing crops. OpenCV thread counts are bounded to avoid CPU oversubscription. Cancellation waits for workers before removing incomplete staging.

Automatic checks use a seeded random 5% of sources, rounded up, with one disjoint native crop per chosen source. All materials remain available for fitting. These internal checks measure learning on familiar sources; fresh photographs and manual displaced-surface reviews judge generalization and usefulness. Dataset approval/exclusion changes metadata only, with an expected index hash and recovery journal. Numeric image values, precision and color tags are not rewritten.

## Full-quality inspection and export

Review opens a `review-manifest.json`, a saved review, or original image maps. `--review /absolute/path/review-manifest.json` and `--dataset /absolute/path/dataset.json` select files at launch. Original dimensions and precision remain intact; screen contrast is a display derivative. **Export Original** copies bytes unchanged. **Export Visible Maps** creates a labeled folder. **Open Copy in GIMP** provides a byte-identical editable copy and keeps the original safe. These actions do not depend on preview decoding finishing.

Every comparison pane keeps its numbered identity above the image: checkpoint run, architecture, training step and checksum when recorded. Reference maps and the untrained flat baseline are labeled separately. Source photo and sample identity remain visible. Labels persist in full-quality pop-outs and export filenames. Older review files expose only the provenance they actually contain.

Studio’s **Generate Material** (⌘R) retains all four maps at the chosen output size. **Open Full Quality** (⇧⌘F) opens the existing selected map immediately. **Export Material** (⌘E) appears in the canvas, toolbar, sidebar, inspector and File menu; it generates pending changes, then saves maps and a Blender script into a new folder. Repeated exports reuse the retained material. Changing only EXR precision re-encodes existing linear data; Float32 exports copy retained image files byte for byte. Pop-out windows keep their files when another material is generated.

HDR color is decoded as a complete float frame before cropping. Finished diffuse pixels are frozen once and reused for preview, export and numeric processing. Exposure matching and smooth highlight rolloff produce an honest 8-bit SDR sRGB PNG; numeric height, normals and roughness receive no gamma conversion. An attached **surface-height** map preserves amplitude through the shared crop and explicit relief controls. Camera-distance interpretation separately enables distance-to-relief processing. See [processing and precision](texture-studio.md#processing-and-precision).

## Saved choices and experimental checkpoints

Studio remembers output size/precision, photo/surface controls, height-source choices, preview selection, inspector visibility and export location. Another photo retains those choices, while an attached map remains photo-specific. The workbench remembers its dataset, native size, crop/map/checkpoint selection, comparison choices and output locations. Review remembers navigation, contrast, visible maps and sample/candidate selection. **Save Decisions & Notes** explicitly saves judgments to the review file. Retired model selections migrate to a supported height choice rather than silently running DINOv2.

Runtime settings locate Python, a working folder and dependencies for legacy comparison/export. DINOv2 is a visual feature encoder, not a pretrained texture-height predictor; its old heads and adapters remain experimental. The untrained scalar head is constant 0.5; the normal baseline is flat with the OpenGL +Y convention. Checkpoint schema, target, step and SHA256 determine identity rather than filenames.

**Export Package** writes a compact inference checkpoint, model card, license/source information, checksum manifest and pinned dependency references. It excludes pretrained encoder weights, source photos and optimizer state. Exporting does not make a retired model production-ready. The Hugging Face panel shows account, destination and visibility inline. **Upload Selected Model** packages the exact selected checkpoint and uploads when clicked, without another confirmation. Destination and visibility persist. Use `hf auth login` and **Refresh Account** if no saved account is available; Swift never displays or stores the token.

Texture processing uses available Metal working-set allowance and machine headroom. Depth cleanup retains source prediction detail through 4K/8K output instead of silently limiting it to 2K. Model input contracts and preview sizes remain separate from final export size. The previous trainer’s saved resource configuration remains available as historical metadata; more memory alone does not create model detail or restore retired training.

Development stays on `main`. Datasets, model weights, app bundles and generated outputs stay outside Git. Enable Git LFS before publishing embedded model weights. Local checkpoint and review availability depends on which files are retained during cleanup.
