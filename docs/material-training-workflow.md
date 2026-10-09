# Native material training workflow

Open a dataset in Dataset Studio, select the material or complete dataset you want to use, and send it to Model Training. Choose height, normal or roughness and the native grid. The app checks that every input and target has the declared dimensions. It references matching originals and creates exact crops when needed; it does not resize or pad training maps. Subjects and crop ancestry govern train/validation separation.

Start with the pinned base, or select a compatible safetensors checkpoint as the warm start. A material adapter updates the selected output branch; developer mode also offers the corresponding decoder. Training executes the complete mapping and gradients in Apple MPSGraph, clips gradient norm and updates Float32 AdamW state. Stop cancels the operation. Stop and Save completes the current update, runs validation and writes a checkpoint. Save Checkpoint queues full validation and a snapshot while training continues. Warm starts restore learned weights and the recorded step; optimizer moments are initialized for the new run.

Source integer maps remain unchanged. Float32 model tensors explicitly divide their decoded codes by the code maximum according to the recorded input/target contract. This model-input conversion is separate from the original arrays and exact crop exports. Normal green-channel conversion is explicit for a declared DirectX source.

Review reference, base and checkpoint maps at full pixels, on a displaced surface and on new photographs. A test photograph is prepared by the same diffuse-processing path used in Texture Studio. Training loss alone does not establish useful detail, generalization or absolute depth. Promotion into Texture Studio remains an explicit selection.

Export includes the separate LoRA. Developer export also includes the complete fused model. Combining adapters requires the same exact base, target, scope and trained layers; weights can be adjusted without changing the original checkpoints. Package checksums cover every weight, configuration and notice file.

Settings hold the working folder, local weights and optional Hugging Face token. No runtime installation or environment setup is required. Model files are downloaded only when requested, and a full checkpoint permits safe removal of the verified downloaded base weights.
