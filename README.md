# Texture Studio

Texture Studio turns surface photographs into diffuse, displacement, roughness and OpenGL normal maps for Blender and other material tools. It provides photo correction, full resolution inspection and Blender export in a native macOS editor.

Train custom models in [Loom Factory](https://github.com/gnaservicesinc/LoomFactory), including depth, roughness, normal and experimental RGB photo cleanup models. [Loom Workshop](https://github.com/gnaservicesinc/LoomWorkshop) trains material LoRAs against the exact base built into Texture Studio.

## Build

Requires Apple Silicon, macOS 26 and Xcode with the macOS Metal toolchain.

```sh
make install
```

This one command builds optimized Release versions of Texture Studio, Loom Factory and Loom Workshop and installs them in `/Applications`. Apps stay closed after installation. Missing Factory or Workshop checkouts are restored from their GitHub repositories. An app with active training can refuse to quit; finish or cancel the run and repeat the command. The installer never force-quits a training run.

`make build`, `make package` and `make smoke` are available for Texture Studio development. `make install DESTDIR=staging` stages all three apps without closing or launching applications; `make install INSTALL_LAUNCH=1` opens all three apps after installation. Installation always uses Release builds.

A build without a trained Texture Loom package supports photo preparation, attached maps and material export. Its model panel clearly identifies the missing built-in model.

Include one complete Loom Factory export when building a deployment:

```sh
TEXTURE_LOOM_PACKAGE="$PWD/exports/Texture-Loom.textureloom" make install
```

The build validates the package inventory, compiles Core ML models and embeds one fixed Texture Loom base. End users cannot replace that base. Building against another exported package creates a new deployment.

## Train photo cleanup models

In Loom Factory, choose a source folder and open **Photo Training**:

- **Repair** trains noise and mild blur cleanup from aligned noisy/clean or blurred/sharp RGB pairs.
- **Lighting** trains uneven illumination correction from photos paired with neutral references.

Choose **Generate Training Pairs…** for a stage-specific synthetic dataset, or import aligned real photo pairs. **Train Repair…** and **Train Lighting…** open the training, checkpoint continuation and saved-model testing controls. See the [photo training guide](https://github.com/gnaservicesinc/LoomFactory/blob/main/Docs/PhotoEnhancerTraining.md).

These are experimental training workflows; qualified pretrained cleanup weights are not included. Review results on unfamiliar materials and clean photos before deployment. Generated microdetail and alpha training remain planned work.

After review, include either exported model folder with `PHOTO_REPAIR_MODEL` or `PHOTO_LIGHTING_MODEL` when building Texture Studio. They are separate from the material package. The editor provides independent toggles and strengths for the bundled stages after framing. See [build and packaging](docs/releasing.md).

## LoRAs

The LoRA panel imports Loom Workshop adapter packages and enables or disables them. An adapter must match the built-in model ID, model version, dataset version and original base weight hash. Each material override includes its actual LoRA weights and the compiled Core ML inference model; incomplete weights-only exports are rejected. Enabled overrides affect inference without modifying the built-in base.

Opening a `.loomlora` export in Texture Studio installs and enables that compatible adapter, then opens its LoRA manager. Loom Workshop's **Install into Texture Studio** action uses the selected application and checks that installation completed.

Photos and original material maps remain unchanged. Application preferences and installed LoRAs live in user storage; the shipped model payload belongs to the signed application build.

## Repositories

- [Texture Studio](https://github.com/gnaservicesinc/Texture_Studio): photo editing, material generation and export.
- [Loom Factory](https://github.com/gnaservicesinc/LoomFactory): datasets, model training, testing and custom deployments.
- [Loom Workshop](https://github.com/gnaservicesinc/LoomWorkshop): paired-map LoRA training for a Texture Studio deployment.
