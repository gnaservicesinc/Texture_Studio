# Texture Studio local validation — October 7, 2026

**Historical first native build:** the model selection and height pipeline below
were subsequently replaced after testing the supplied surface photographs.
Apple Small V2 and portrait-depth material relief are retired. Current behavior
uses DA3-GIANT-1.1 with depth-derived height; see the [workflow](texture-studio.md)
and the [current surface-validation record](texture-surface-validation-2026-10-07.md).

Validated on an Apple M2 Max Mac with 64 GiB unified memory, full Xcode 27.2,
and macOS 27.2. These are local checks; no remote CI, publication, or
notarization is claimed.

## Native application and retained tools

- Full hosted XCTest suite: **37 tests passed, zero failures**. Coverage:
  geometry/exports 7, decision service 10, model lifecycle 11, photo evidence 4,
  full spatial fusion 3, workspace state 2.
- Retained Python extraction/research tools: **318 tests passed**.
- The canonical `script/build_and_run.sh --verify -- --open …` built Debug,
  launched the standalone application, and verified the running executable.
- Final Release build and package passed signature, arm64, minimum macOS 26,
  Apple system dependency, and loopback networking checks. The archive is
  `dist/Texture-Studio-macos-arm64.zip`; its SHA-256 is
  `f580817a06e61741ffb35e60117c2f72b8d7955c0c38ae3ad1a4f522c5760a11`.
- A real 5712 × 4284 iPhone HEIC imported camera metadata, embedded disparity,
  and Apple/ISO HDR gain-map information. Both EXR storage paths and all four
  map dimensions were checked in native smoke runs.
- FLOAT32 EXR sample values and image orientation round-trip exactly in the
  regression fixture. HALF exports are checked for actual HALF channel types;
  a requested 32-bit file is checked for actual FLOAT channels.
- An independent nonflat Metal normal probe matched the analytic OpenGL +Y
  normal within `2.98e-8` maximum channel error.

## Spatial-photo evidence

Complete Vision registration and blending tests use textured synthetic
surfaces, not just an isolated patch classifier. A half-resolution companion
with different exposure/color contributes actual pixels. A translated view
contributes only where registered evidence agrees, leaving uncovered edges
exactly equal to the primary photo. An unrelated surface contributes nothing.

These checks caught and fixed two bugs: a scalar mask that made blending a
no-op, and a cropped warp extent that changed the coordinate frame. Synthetic
checks establish functional behavior, not visual quality on real spatial
material captures or general nonplanar reconstruction.

## Local models

Apple's revision-pinned Core ML Depth Anything V2 Small package was downloaded,
SHA-256 verified, compiled, and used for native inference. Managed installation,
removal, missing/moved paths, external unlinking, and output selection have
regression coverage. Its fixed 518 × 392 FP16 output is a relative shape guide;
FLOAT32 export does not increase its source precision.

The local Ollama server reports **0.40.0**. The exact **clef:27b-nvfp4** download
completed, with these actual local metadata values:

| Property | Observed value |
| --- | --- |
| Format / quantization | safetensors / nvfp4 |
| Runner | mlx |
| Capabilities | decision, vision |
| Download size | 18,107,132,722 bytes |
| Model memory reported during inference | 18,138,724,872 bytes, approximately 16.89 GiB |

A native request returned a proposal accepted by the strict fixed-choice
decoder. After `keep_alive: 0`, `/api/ps` reported an empty model list. Weights
remain installed for the user's app. This confirms local MLX execution and
unloading, without establishing a comparative speed or material-quality claim.
The supplied transform screenshot is a runtime fixture, not a surface-quality
benchmark. A second request using the real HEIC portrait at 768 × 576 returned
accepted choices in 12.691 seconds including model reload, reporting 1,101
input tokens and zero output tokens. That single runtime check is not a latency
or material-quality benchmark. Ollama's model context defaults and actual computation remain server
controlled; the app bounds the image, request duration, and accepted token
usage rather than sending unsupported generation controls.

## Review limits

The computer-use tool could not start its native pipe, and Screen Recording
permission was not granted. Interactive GUI inspection was therefore not
completed; build, launch, native processing, and export checks did complete.

Roughness and photo-derived height remain editable estimates. Hard-shadow
reconstruction, seamless tiling, and a trained material-specific refiner are
not implemented. The available real HEIC is a portrait, so actual surface
photos still need comparison under neutral and grazing Cycles lighting.
Source photographs and original scientific extraction data remain unchanged.
