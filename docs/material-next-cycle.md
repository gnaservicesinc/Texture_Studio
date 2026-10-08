# Next material-fitting cycle

This source tree contains the implementation and documentation. Experiment checkpoints, generated maps, source snapshots, recovery recipes and reports are retained locally and are not included. Record names below identify that local archive.

**Current workflow:** use [the material quality workflow](material-quality-workflow.md) for native 2K preparation, resumable training, and photo-on-displaced-geometry review. The historical fitting scores below diagnose the training objective; they do not rank realistic material quality. A candidate can be more useful despite greater error against a particular reference.

The current frozen batch has **97 materials and 291 native 1024 crops: 194 training corners and 97 disjoint known-material validation regions**. It adds 28 complete materials and 84 crops to the historical 69-material batch. The original 207 sample manifests, 882 crop PNGs and prior index entries remain unchanged. Complete original diffuse, displacement, normal and roughness sets should retain their actual source precision; preparation handles folder names, native crops and metadata.

The source records identify AmbientCG `Snow013`, `Snow014` and `Snow015` separately from Poly Haven `snow_01` and `snow_02`. Their original PNG files matched all 12 full members of the official packages. Each official asset page lists **Surface Photogrammetry** and CC0: [Snow013](https://ambientcg.com/view?id=Snow013), [Snow014](https://ambientcg.com/view?id=Snow014), [Snow015](https://ambientcg.com/view?id=Snow015). Package/member hashes, provider IDs, original filenames and page URLs are recorded. This capture-method declaration does not establish physical accuracy of individual displacement samples. Temporary verification ZIPs were removed.

At the user's request, `brick_4`, whose displacement is 8-bit, was removed from the active source set and moved to macOS Trash with a full checksum record. It was never in the historical 207-sample index. Preparation now excludes 8-bit displacement; 8-bit diffuse, normal and roughness maps remain valid at their real precision. `leaves_forest_ground` is incomplete and remains untouched outside the index. The frozen batch deliberately excludes later source arrivals.

## Completed matched adapter comparison

`diagnose_material_adaptation.py` completed **1,200 MPS updates per variant**, giving stucco, soil, cotton and brick exactly 300 balanced updates per native crop. Both variants began with the identical checked native head previously trained 300 times on stucco; this warm-start bias is explicit. Supervised RGB/height stayed native 1024, while only the contextual encoder used 518. Raw UInt16 height codes became Float32 in active batches without gamma, range stretching or a target resize.

The selected frozen control scored **1.545287** at step 1200. Selected rank-8 LoRA scored **1.571060** at step 800, **1.668% worse**; its final step 1200 scored 1.624575. Runtime was 203.97 versus 419.33 seconds, with sampled Metal driver peaks of 2.201 versus 5.430 GB. Actual adapter gradients/updates, identical initial predictions and unchanged pretrained/source/target fingerprints were verified. The locally retained quality review (`four-material-adaptation-01/quality-review.json`) and locally retained fixed-gain preview (`four-material-adaptation-01/final-fixed-gain-comparison.png`) distinguish final weights from selected checkpoints.

Both variants fit the repeated training crops well, but brick-region amplitude remains weak and fine stucco structure is softened in the existing map previews. The metrics show different tradeoffs between materials; whether LoRA adds useful visible detail needs a consistent rendered-material review. No experimental checkpoint is selected in Texture Studio. This four-material experiment did not train on the newly added families and did not test fresh materials or raw camera photographs.

## Next controlled fitting step

Inspect the relief represented by each material's two prepared training corners before adding more model parameters. The locally retained raw corner inspection (`four-material-adaptation-01/corner-height-statistics.json`) found brick height standard deviations of 0.022545, 0.128480 and 0.209493 in training corners 001/002 and validation 003, respectively. The second training corner contains deeper joints/cracks, so both corners should receive equal sampling while 003 stays held out. Use a frozen encoder/native head as the control, retain all target amplitudes and detail, and expose it to representative relief across the prepared regions. Measure fitting separately from transfer to a disjoint known-material region. Additional photographed brick/ground families can then enter a balanced fit. Increasing adapter rank or encoder size is not supported as the immediate remedy by this trial.

Published 2K originals still require a versioned surface-footprint split before training: the current 1K corners from the four available 2K parents overlap their 4K center validation regions. Cross-resolution registration remains unverified. Keep provider-original resolution variants rather than resizing training targets locally.

For an intentional reproduction of the completed four-material comparison, choose a new output directory; existing runs are never overwritten. Replace `/path/to/checkpoint.pt` with an existing schema-compatible frozen-feature material-head checkpoint from your own training or local experiment archive. Reproducing the historical comparison requires its identical starting checkpoint and source snapshot; choosing another compatible checkpoint starts a new comparison. The source publication contains no trained starting weights:

```sh
.venv/bin/python scripts/diagnose_material_adaptation.py \
  --dataset /opt/ipde/material-dataset \
  --head-checkpoint /path/to/checkpoint.pt \
  --output out/material-training/four-material-adaptation-reproduction-02 \
  --updates-per-crop 300 --evaluate-every 400 --checkpoint-every 100 \
  --max-minutes 12 --device mps --allow-unreviewed --mask-transparent-input
```

The exact pinned pretrained Base and official code were cached in the experiment workspace; a new machine must obtain and verify those pinned dependencies before running the comparison. Checkpoints contain only locally trained head/adapter weights and optimizer state, with no duplicate pretrained Base. Model/optimizer states are saved atomically every 100 updates. The checked initial candidate remains eligible if updates worsen the declared objective. Partial schedules are labeled incomplete; automatic CLI resume is not implemented.


The current compact recovery recipe is `/opt/ipde/material-dataset/recreation-97-materials-parent-note-refs.json.gz`, also preserved in the local `source-refresh-97/` archive. It is **319,073 bytes**, with SHA256 `f84eb5f1f5d2e178dbbc92f6953471f62c451071aa33a19b8f13b8b64962005a`. All 415 required parents have verified download origins, using 403 distinct downloads. The two PNG files named `.txt` are restored from exact parent references; the unique 492-byte README remains embedded unchanged. A real local restoration matched every original note byte and a second restoration was idempotent. The earlier full 97-material recipe and historical 56-/69-material recipes remain unchanged.

After obtaining and verifying parent maps in a separate source directory, restore source notes before regenerating the recorded crops:

```sh
.venv/bin/python scripts/material_recreation.py restore-notes \
  --recipe /opt/ipde/material-dataset/recreation-97-materials-parent-note-refs.json.gz \
  --sources /path/to/recovered/sources
```

The command checks parent hashes/sizes and existing destinations, preserves original filenames and refuses to overwrite differing files. It uses no network and also accepts older recipes with embedded source notes. Full source-map downloading and crop rebuilding remain the explicit recovery steps in the recipe.
