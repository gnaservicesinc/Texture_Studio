# Classifier integration and specialist workflow validation

Validated Texture Studio 0.9.20, build 115, on macOS arm64. The user's exported
Diffuse Texture Type Classifier IFPv2 is bundled as a native Core ML classifier.
The original export and bundled source model have SHA-256
`e2936fcef0781fb4f85eb7e879cc2f7d865528415180d894afd74eddc1e048be`.
The user's Create ML project and classifier folder are preserved outside the
committed application resources.

## Live preparation and preservation

- Installed-app refresh prepared 375 original material sets across 122 families
  and 15 classifier categories, with no warnings. Fourteen categories have paired
  height, roughness and normal targets; Other Diffuse Textures has no paired
  targets yet. No specialist model is automatically approved by preparation.
- The main Chimpmonk dataset remains the source database; additional source-folder
  pairs are referenced by specialist datasets without adding them to its
  membership. Ten additional sets represent Paper003, Paper005, Cardboard_01 and
  Cardboard002. AmbientCG and authored role filename formats both load.
- All 147 classifier images present before this task still exist at their original
  paths with identical SHA-256 hashes. Refresh added Cardboard002 under Paper,
  producing 148 classifier images across 15 populated labels. It retained 24
  classifier-only examples without fabricating reconstruction targets.
- Apple's `MLImageClassifier.DataSource.labeledDirectories.labeledImages` loaded
  all 148 images from the Create ML training view. The view's bytes and image
  dimensions match the curated originals. Empty default labels are excluded,
  preventing the prior missing-data error.
- Cardboard002's current image classifier prediction is Leather, approximately
  79% confidence. Its curated Paper label overrides specialist training membership;
  Paper contains ten sets and Leather contains six. Retraining the classifier is
  still required to improve automatic image-based routing for this material.
- Classifier resizing is isolated to the Core ML image adapter. Reconstruction
  datasets retain exact source dimensions and native pixel precision, including
  grayscale and 16-bit files. Training crops are not resized; optional flips and
  quarter-turn rotations remain Trainer settings. Surface-map values use the
  existing linear Float32 inference/export path.

## Focused native verification

All listed runs passed with zero failures. Scopes overlap and are not an aggregate
count of unique tests.

| Run | Tests | Coverage |
| --- | ---: | --- |
| Dataset and classifier discovery | 63 | Native material discovery, paired-map filenames, dynamic class folders, refresh preservation and classifier datasets |
| Runtime and workflow | 40 | Core ML execution, automatic routing, grayscale/16-bit PNG handling, filename variants, prepared-image reuse, fallback, recipes and workflow state |
| Final store regressions | 72 | Specialist handoff and relaunch settings, comparison, direct interaction, preferences, training completion and shared Studio workflow |
| Final specialist service and router | 14 | Refresh ownership and preservation, subtype expansion, membership tracking, checkpoint provenance, approval and automatic routing |

The final service/router run included a genuine native 256-pixel compact training
update, saved checkpoint, provenance registration, explicit approval and routing.
Changing a selected category's source invalidates its model; changing an unrelated
category does not prevent a valid category from routing. This verifies execution
and lifecycle behavior, not novel-material generalization or production quality.

Build-tool regression checks passed, including the classifier payload allowlist
and rejection of unregistered model/checkpoint payloads. `git diff --check` passed.
Detailed transient logs are `/tmp/texture-studio-filename-tests.log`,
`/tmp/texture-specialist-workflow-tests.log`,
`/tmp/texture-specialist-store-regression-tests.log`,
`/tmp/texture-specialist-tests.log` and
`/tmp/texture-specialist-build-tool-tests.log`.

## Package and installed application

`make package` succeeded and produced `dist/Texture-Studio-macos-arm64.zip` with
its SHA-256 sidecar. Native-bundle checks and strict signature verification passed
for Texture Studio and all four bundled material tools. The model payload guard
allows only the registered compiled classifier among bundled ML model payloads.

After the user saved and closed Texture Studio, the native installer installed
build 115 at `/Applications/Texture Studio.app`. The installed bundle is byte for
byte identical to the packaged build. Installed native-bundle, payload and strict
signature checks passed, including all four material tools. The installed smoke
test returned exactly:

```text
Texture Studio native smoke passed
```

Installed-app specialist refresh completed successfully. Its final registry and
summary are recorded in `/tmp/texture-specialists-installed-final.json` and
`/tmp/texture-specialists-installed-final-summary.json`. This validation does not
claim an interactive GUI click-through or trained production specialist quality.
