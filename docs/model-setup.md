# Local material model setup

The material workbench uses a local Python/PyTorch backend and a complete revision-pinned PBRnxt base. The base's source, tensor layout and checksum are checked before loading. No different architecture or base checkpoint is silently substituted.

Install or locate the base from the Model Training workspace. A saved full material `.safetensors` checkpoint can also become the starting base. Normal-mode refinement saves a separate LoRA tied to the exact base; Developer mode additionally exports the full model with that adapter merged.

```sh
python scripts/material_model_workbench.py install-base --destination /path/to/pbrnxt-base
python scripts/material_model_workbench.py capabilities --memory-gib 32 --cache-gib 1 --scope final-map
```

The installation command downloads the pinned upstream source and weights. Training and inference use local files. Python needs the repository's declared dependencies, including PyTorch, NumPy, unchanged-depth OpenCV decoding and OpenEXR. HEIF extraction separately requires `pillow-heif>=1.5.0`.

Use **Refresh Account** to read a saved Hugging Face login. Upload packages only the selected material model, its model metadata and dependency/license files; source photos and optimizer state are excluded. Successfully uploaded models are recorded in the app's Hub catalog and offer a **Download** action using the recorded repository revision and checkpoint identity.

App-owned downloaded bases can be removed. Located external bases are unlinked without deleting the user's files. Retain a full checkpoint or a recorded download origin before removing a base needed by an adapter. A separate LoRA needs the exact matching base again for inference or further refinement.

Inference consumes a prepared diffuse PNG. A photo dropped into the native app first receives the shared geometry, color and illumination preparation. The backend rejects `--input-kind photo` so an untreated image cannot be mistaken for a valid material-model test.

All model exports use `.safetensors`. Full checkpoints and LoRAs are distinct formats with explicit architecture, target, base and module metadata; current readers do not implement compatibility for removed development experiments.
