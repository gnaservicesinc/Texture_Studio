# Material training workflow

The native workbench uses `material_workbench.py` for source-referenced dataset curation and `material_model_workbench.py` for material LoRA training, inference, export and Hugging Face recovery. The backend is a complete pinned PBRnxt network adapted to a native output grid.

1. Open the source dataset. Its small index references complete original maps; source images remain in place. A material's 2K, 4K and 8K files form separate registered resolution sets.
2. Select height, roughness or normal, then a training size supported by the original pixels and hardware. Smaller files are excluded from a larger training grid: a 1K color or numeric map never enters a 2K run.
3. Review the full selected grid. Exact-size originals are referenced directly. Larger originals supply explicit native crops: one centered crop from a 4K set, or three corners from the four available corner choices in an 8K set. At 2K, every input, numeric target and registered color variant is literally 2048×2048; preparation never downscales or pads the source. A changed crop is saved once in the temporary training dataset and reused.
4. Select the whole dataset or a particular material, LoRA scope and run limits. Automatic validation groups related resolution sets and color variants by original asset family. A held-out family cannot contribute another resolution or color to training. When a validation family has only one included 8K resolution set, two disjoint corner crops train and the third checks the result. This measures new regions of a known material; it does not establish unseen-material generalization. A centered single-region fit has no independent region check.
5. Run training. The displayed complete diffuse crop and registered numeric target are the tensors supplied to the model. Legacy `col`, numbered `col` and other registered color variations can share the same numeric target: each update randomly selects a real color variant using the run seed. All variants must have the same crop, source registration, dimensions and color transfer. Validation and base/refined comparisons use a deterministic variant and record its actual path, hash and identity. No fabricated photographic exposure/color augmentation or hidden crop is used. Height, roughness and normal targets retain their linear numeric codes through the declared integer-to-Float32 transfer.
6. Inspect base, reference and trained predictions on the same pixels. Save accept/exclude decisions and notes. The local decision model may preselect a recommendation based on detail, visual appeal and artifacts.
7. Export the LoRA. Developer mode also merges it with the selected base and saves a full `.safetensors` checkpoint, while retaining the separate LoRA. Compatible adapters can be combined with explicit weights.
8. Upload the selected package when desired. Developer mode exposes upload after training. The saved Hub catalog offers later download by recorded repository revision.

Completion or Stop saves the current adapter and removes owned temporary training files. A failure after an update retains a recovery adapter while purging those temporary inputs. Per-size review metadata remains; source maps and saved checkpoints are retained. A preview after cleanup reconstructs the literal recorded crop from the original source. Extended runs retain a bounded live update summary and append every actual input identity to `updates.jsonl`, avoiding repeated rewrites of an ever-growing history. Remove an app-owned base only after retaining an appropriate full checkpoint or a recoverable download origin.

Older sets without native 16-bit displacement can still train their available roughness or normal target. They are skipped for displacement training. Specular and other unsupported auxiliary material maps do not silently replace any target.

A test photograph is first leveled and prepared as a diffuse map through the same path used by the final application. The Python inference command accepts a diffuse PNG and rejects `--input-kind photo`; the UI performs the photo preparation. This keeps tests consistent with what the material model was trained to receive.

The supported preparation grids are 256, 512, 1024 and 2048, subject to source and hardware limits. Larger generation/export sizes remain independent. Automatic memory selection recommends 24 GiB for 1K or 48 GiB for 2K final-map LoRA on this 64 GiB M2 Max with the default 0.5 GiB decoded-pair cache. Developer mode exposes a manual override. Changing the training size recalculates the allowance rather than retaining the previous grid's budget.

Full float32 2K displacement LoRA completed a real pretrained-model forward, backward and AdamW update with a sampled Metal driver peak of 42.3448 GiB. Three consecutive 1K updates also completed, sampling 19.7359 GiB. Outer RRDB activation checkpointing recomputes exact operations during backward, preserving weights, forward values and pixel precision while reducing retained activation storage. The wider 2K map-decoder scope remains unqualified and excluded by its conservative memory estimate on this hardware. [Measured qualification](material-lora-memory-qualification.json) records the budgets and limits. These synthetic probes establish execution and memory behavior; they do not establish material quality or extended-run stability. Resource estimates are not a guarantee about every machine's peak usage.

Useful read-only commands:

```sh
python scripts/material_workbench.py dataset --dataset /opt/ipde/material-dataset
python scripts/material_model_workbench.py capabilities --memory-gib 48 --cache-gib 0.5 --scope final-map
```

To prepare an explicit temporary training view, use `prepare-size --dataset DATASET --size SIZE --automatic-validation`. Its response identifies the prepared dataset and authoritative source dataset. `cleanup-size --dataset PREPARED_DATASET` removes only app-owned temporary data and preserves the size's review fields. The native workbench performs these actions around training automatically.

See [data contract](material-training-data.md), [native tools](native-material-tools.md), and [quality acceptance](material-model-vetting.md).
