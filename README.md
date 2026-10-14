# Texture Studio

Texture Studio turns surface photographs into diffuse, displacement, roughness and OpenGL normal maps for Blender and other material tools. The macOS texture editor, photo correction, full resolution inspection and Blender export remain in this repository.

Model development belongs to [Loom Factory](https://github.com/gnaservicesinc/LoomFactory). Texture Studio contains no dataset manager, classifier trainer, full model trainer or checkpoint comparison application. [Loom Workshop](https://github.com/gnaservicesinc/LoomWorkshop) provides the separate guided LoRA training workflow using paired material maps and this build's fixed base.

## Build

Requires Apple Silicon, macOS 26 and Xcode with the macOS Metal toolchain.

```sh
make install
```

This one command builds optimized Release versions of Texture Studio, Loom Factory and Loom Workshop and installs them in `/Applications`. Apps stay closed after installation. Missing Factory or Workshop checkouts are restored from their GitHub repositories. An app with active training can refuse to quit; finish or cancel the run and repeat the command. The installer never force-quits a training run.

`make build`, `make package` and `make smoke` remain available for Texture Studio development. `make install DESTDIR=/path/to/staging` stages all three apps without closing or launching applications; `make install INSTALL_LAUNCH=1` explicitly opens all three apps after installation. A `CONFIGURATION=Debug` override cannot change the production configuration used by `make install`.

A build without a trained Texture Loom package supports photo preparation, attached maps and material export. Its model panel clearly identifies the missing built-in model.

Include one complete Loom Factory export when building a deployment:

```sh
TEXTURE_LOOM_PACKAGE="/absolute/path/Texture-Loom.textureloom" make install
```

The build validates the package inventory, compiles Core ML models and embeds one fixed Texture Loom base. End users cannot replace that base. Building against another exported package creates a new deployment.

## LoRAs

The LoRA panel imports Loom Workshop adapter packages and enables or disables them. An adapter must match the built-in model ID, model version, dataset version and original base weight hash. Each material override includes its actual LoRA weights and the compiled Core ML inference model; incomplete weights-only exports are rejected. Enabled overrides affect inference without modifying the built-in base.

Opening a `.loomlora` export in Texture Studio installs and enables that compatible adapter, then opens its LoRA manager. Loom Workshop's **Install into Texture Studio** action uses the selected application and checks that installation completed.

Photos and original material maps remain unchanged. Application preferences and installed LoRAs live in user storage; the shipped model payload belongs to the signed application build.

## Repositories

- Texture Studio: this repository and `/opt/ipde/ipde`.
- Loom Factory: `/opt/ipde/ipde/LoomFactory`, a fresh project for fully custom Texture Loom deployments.
- Loom Workshop: `/opt/ipde/ipde/LoomWorkshop`, a fresh paired-map LoRA trainer, developed after the Factory and Studio contracts.

The retired integrated training implementation is preserved in Git history rather than linked into the new products.
