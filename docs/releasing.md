# macOS releases

Texture Studio is a native Apple Silicon application. Model development and LoRA training are provided by the independent Loom Factory and Loom Workshop applications.

Run `make package` for an optimized Release build and the ZIP in `dist`. Set `TEXTURE_LOOM_PACKAGE` to a complete Loom Factory `.textureloom` export when building a deployment with learned material inference. The integration verifies every registered artifact, compiles Core ML models and embeds the fixed model payload before signing.

Run `make install` at the repository root to build and install Texture Studio, Loom Factory and Loom Workshop. Apps remain closed by default; set `INSTALL_LAUNCH=1` to open them after installation. Missing companion checkouts are cloned from their GitHub origins. All three Release builds and signatures are checked before installed applications are asked to close. Active training can refuse that request, leaving installed applications untouched. The suite never force-quits an app, and each installer stages and verifies its replacement with rollback on failure.

Installation does not create distribution ZIPs or run Debug builds. `CONFIGURATION=Debug` is ignored by this production install target. Use `DESTDIR=staging` to install all three into a staging tree without closing or launching applications. Alternate Factory and Workshop checkout paths can be supplied as `LOOM_FACTORY_REPO` and `LOOM_WORKSHOP_REPO`.

## Optional photo cleanup models

Train and review Repair and Lighting exports in Loom Factory's **Photo Training** workspace. Each exported folder contains `enhancer.json`, `Enhancer.mlmodel` and `weights.safetensors`. Include reviewed exports at build time:

```sh
TEXTURE_LOOM_PACKAGE="$PWD/exports/Texture-Loom.textureloom" \
PHOTO_REPAIR_MODEL="$PWD/exports/repair" \
PHOTO_LIGHTING_MODEL="$PWD/exports/lighting" make install
```

Use your own export locations. Either photo model is optional. The build validates its product, architecture, native interface and checksums, compiles Core ML, and embeds the model before signing. Omitting a photo model removes any stale copy from a previous build. Photo models do not supply the material classifier or depth, roughness and normal models.

The current photo interface accepts opaque SDR 1024 × 1024 inputs. The editor applies enabled Repair and Lighting stages after framing, with independent strengths. These are experimental products; training and export do not qualify their appearance. See [Factory's training and evaluation guide](https://github.com/gnaservicesinc/LoomFactory/blob/main/Docs/PhotoEnhancerTraining.md).

The current local build is ad-hoc signed. Signing with an Apple Developer identity and notarization are separate distribution steps.
