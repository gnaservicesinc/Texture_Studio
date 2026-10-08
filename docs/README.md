# Texture Studio documentation

The current desktop app is **Texture Studio**, a standalone SwiftUI app for
photo-based Blender materials. Start with the repository README and
[building and releasing](releasing.md).

The [native material tools](native-material-tools.md) provide full-detail map review,
checkpoint comparisons, crop curation, native 1K/2K training and explicit model export.

See the [surface workflow](texture-studio.md) and [material refinement research](material-refinement-research.md)
for the DA3 backend, depth-to-height conversion and Poly Haven/MatSynth training options.
The [surface validation record](texture-surface-validation-2026-10-07.md) reports actual
supplied-photo inference, EXR checks and Blender Cycles renders.
The [training data contract](material-training-data.md) defines original 16-bit targets,
curated native-resolution crops and separate disk/compute precision.
The [automated preparation and training workflow](material-training-workflow.md)
provides the crop, naming, metadata, verification and native-height pilot commands.

The [legacy documentation](legacy/README.md) records the retired Qt studio
interfaces. It does not describe the native app or its build requirements.
Scientific extraction and training audits remain useful references for the
retained Python command-line tools; those tools do not communicate with Texture
Studio automatically.
