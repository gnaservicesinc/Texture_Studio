# Texture Loom deployment and adapters

Loom Factory, Texture Studio and Loom Workshop use two versioned package formats. Both are ZIP archives with `manifest.json` at the root, relative artifact paths and SHA-256 file inventories. Source images and credentials are excluded.

| File | Schema | Consumer |
| --- | --- | --- |
| `.textureloom` | `texture-loom.package.v1` | Texture Studio's build process |
| `.loomlora` | `texture-loom.adapter.v1` | Texture Studio's LoRA importer |

The base package has `id`, `name`, `version`, `classifier`, `categories` and `files`. The classifier contains native Core ML, its declared input and output names, native input dimensions, and dataset labels with IDs, names and `sourceLabel` values. A Create ML classifier needs no Safetensors weights. Its actual class labels must exactly match the package catalog, and every label must route to an included category with material models. Each category has an ID, name, version and models for `depth`, `roughness` or `normal`. Material-model entries include `format: coreml`, `path`, `weightsPath`, architecture and the original weights' SHA-256. Architecture descriptors accompany these weights so Workshop can train compatible adapters.

A classifier can accept a native RGB image feature (`inputType: image`) or a float32 RGB tensor (`inputType: multiArray`, `inputShape: [1,3,height,width]`). It outputs its original string label through its declared output name; consumers map that label to the fixed dataset ID through `sourceLabel`. Model-specific RGB or BGR preprocessing remains inside Core ML. Material-model inputs are float32 RGB tensors named `rgb` in NCHW layout, with sRGB values in `[0,1]`. Map models accept flexible dimensions from 32 to 8192 pixels and return `material_map`: one channel for depth and roughness, three OpenGL channels for normals. Targets preserve their native code values and precision during training. The classifier is sealed with these material models and cannot be replaced independently in Texture Studio or Workshop.

Texture Studio's build validates the archive, compiles the native models, updates their registered paths and inventory, and seals the fixed base in the signed app. To include an export:

```sh
TEXTURE_LOOM_PACKAGE="/absolute/path/Texture-Loom.textureloom" make install
```

An adapter has its own ID, name and version, `baseModel` with the exact base ID and version, and matching category IDs and versions. Every map override includes a Core ML inference model, actual LoRA Safetensors weights and `baseSHA256`. Workshop freezes the installed base during training and exports a fused native inference copy. Importing never modifies the bundled weights.

Opening `.loomlora` in Texture Studio validates, installs and enables the adapter, then shows the LoRA manager. Workshop targets the selected installed application and verifies the resulting adapter receipt before reporting installation complete. Disabling a LoRA returns inference to the bundled model. Only one enabled override applies to a given dataset and map at a time.

Packages reject outside paths, symbolic links, missing or changed registered files, incomplete native model inventories, unknown datasets and mismatched base versions or weight identities. A successful training/export/integration check proves execution; deployment quality requires review on unseen materials.
