# macOS releases

Texture Studio is one native Apple Silicon application. The retired nested material dataset, trainer, checkpoint and review apps are excluded from every package.

Run `make package` for an optimized Release build and the ZIP in `dist`. Set `TEXTURE_LOOM_PACKAGE` to a complete Loom Factory `.textureloom` export when building a deployment with learned material inference. The integration verifies every registered artifact, compiles Core ML models and embeds the fixed model payload before signing.

Run `make install` at the repository root to build, install and open Texture Studio, Loom Factory and Loom Workshop. The other two applications remain independent repositories; missing local checkouts are cloned from their GitHub origins. All three Release builds and signatures are checked before installed applications are asked to close. Active training can refuse that request, leaving installed applications untouched. The suite never force-quits an app, and each installer stages and verifies its replacement with rollback on failure.

Installation does not create distribution ZIPs or run Debug builds. `CONFIGURATION=Debug` is ignored by this production install target. Use `DESTDIR=/path/to/staging` to install all three into a staging tree without closing or launching applications. Set `INSTALL_LAUNCH=0` for a normal install without relaunching. Alternate Factory and Workshop checkout paths can be supplied as `LOOM_FACTORY_REPO` and `LOOM_WORKSHOP_REPO`.

The current local build is ad-hoc signed. Signing with an Apple Developer identity and notarization are separate distribution steps.
