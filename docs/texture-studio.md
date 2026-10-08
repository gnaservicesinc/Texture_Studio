# Native Texture Studio workflow and model choices

The goal is a useful material derived from the user's photo. Source files remain unchanged. The standalone SwiftUI app does not require an IPDE Studio project or its IPC/session machinery. JSON recipes and Blender exports provide file-based interoperability.

## Geometry and supporting views

Transforms use a camera-centered pinhole projection of a plane under X/Y/Z rotation. Metadata supplies an approximate focal length when available; a user override is provided. EXIF does not identify the surface normal, so controls express the intended correction. A maximum square crop is solved against the projected convex footprint with a sampling inset; tighter framing stays inside it. X=7.93°, Z=0.22° is supported. Nonplanar surfaces are approximated by a plane.

Apple's [archived camera reference](https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/Cameras/Cameras.html) describes older device capabilities, rather than a current calibration database. Actual metadata and [ImageIO spatial groups](https://developer.apple.com/documentation/imageio/writing-spatial-photos) take precedence over device-name assumptions. Gain maps and mattes are separate from photographic views; partially transparent mattes are not stretched into primary material detail.

The primary photo anchors spatial fusion. [Vision homography](https://developer.apple.com/documentation/vision/vnhomographicimageregistrationrequest) aligns companions on bounded grids, then exposure and per-channel color are fitted using unclipped samples. Textured patches must agree after matching. Accepted contributions are capped; occlusions, parallax and weak evidence keep the primary. Native companion images are warped for final sampling. This is conservative enhancement rather than measured stereo reconstruction or a claim of super-resolution.

## Processing and precision

Core Image color processing uses extended linear sRGB and a Metal float context. HDR gain-map color is decoded as a complete float frame before transforms or cropping, then exposure is matched to the base photograph. Delighting divides by broad illumination with bounded gain and uses a smooth highlight rolloff for the final SDR diffuse. Noise reduction is deliberately modest. The finished diffuse is materialized once; preview, PNG export and downstream numeric processing use the same frozen pixels, avoiding exposure changes or red tiles from repeated lazy HDR evaluation.

Depth/maps use a separate color-unmanaged float context. Numeric values receive no color-profile gamma or photographic tone mapping. Roughness is an editable local-contrast estimate. Camera distance maps use global plane removal, depth-only median/bilateral cleanup and a declared artistic relative range. An attached map explicitly identified as **surface height** bypasses that distance normalization and cleanup, preserving its supplied amplitude through registration, the shared crop and explicit relief controls. Normals are mathematical central differences of the final height, scaled by material width and displacement amplitude, with OpenGL +Y.

Embedded portrait depth is never used as material relief, even when an older recipe requests it. Photographic brightness relief is an explicit artistic control that defaults to zero; albedo and shadow patterns therefore do not create bumps by default. With no depth selected, displacement and normals stay neutral. Old recipes migrate away from the retired model and embedded-depth mode.

**Protect near-flat surfaces** reduces the artistic height amplitude when a positive distance prediction has less than 5% residual depth contrast relative to its median distance. This avoids stretching tiny model variations into large bumps. It is an adjustable safeguard, not calibrated roughness detection. Explicit surface-height maps bypass it; attached maps have a selectable surface-height versus camera-distance interpretation. Raw samples remain unchanged.

Source bit depth, model precision, working precision and export precision are distinct. Diffuse is honestly 8-bit sRGB. Apple's [OpenEXR writer](https://developer.apple.com/documentation/coreimage/cicontext/writeopenexrrepresentation(of:to:options:)) was verified to emit HALF channels, so float32 export uses standard uncompressed FLOAT scanlines with bounded tile/row buffers. Both paths inspect actual channel types. Staging and new-folder export protect previous materials.

Use Blender **Non-Color** for height, roughness and RGB normal maps. Its [Normal Map node](https://docs.blender.org/manual/en/4.1/render/shader_nodes/vector/normal_map.html) handles the encoded tangent-space map; GIMP is not required. The generated script defaults to full normal relief. Its `USE_GEOMETRIC_DISPLACEMENT` option connects geometric displacement and disables that equivalent normal contribution to avoid applying the same height twice. Cycles subdivision/displacement settings remain explicit user decisions.

## RAFT and depth models

[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) estimates correspondence between two views and remains outside the active material workflow. Imitating monocular teacher depth has not demonstrated better material relief here. The earlier small pilot did not establish a reliable texture model.

[DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) remains an explicit camera-depth option using PyTorch/MPS and the exact revision-pinned Float32 checkpoint. Inference resolution is selected independently of export size, with resource checks and cancellation. The Apple Small V2 Core ML conversion is retired. Core ML remains an optional custom backend rather than a format requirement.

DA3's published model is a relative scene-depth/geometry model, not a texture-height estimator trained on material displacement. A 32-bit file or an 8K resample cannot add details absent from its prediction. Its weights are CC BY-NC 4.0; commercial use requires permission or a differently licensed model. DINOv2 material heads are retired and cannot be activated or resumed. The **Model Training** workspace currently prepares paired data and supports legacy review, comparison and export. A replacement backend is not connected. See [current model vetting](material-model-vetting.md).

Depth Pro can be visually strong at object boundaries, but its [weight license](https://huggingface.co/apple/DepthPro/raw/main/LICENSE) limits use to research and explicitly excludes product development. It is not bundled or offered as the production download. External map compatibility does not itself establish licensing permission.

[NAFNet](https://github.com/megvii-research/NAFNet) width32 denoising is a compact restoration candidate to benchmark with PyTorch/MPS or a validated Core ML conversion. [Restormer](https://github.com/swz30/Restormer) has first-party restoration weights and is another candidate. Neither is currently integrated. Evaluate preservation of grain, repeated texture, edge halos and final Cycles appearance before adding them.

Candidate experiments include a texture-specific normal-to-height baseline and a supervised material refiner with a depth prior. They must preserve the photographed pattern and native map detail. LoRA and optimizer choice do not establish suitability by themselves. Full-resolution real-photo maps and displaced Blender surfaces must pass visual acceptance before any replacement becomes a Studio model; Core ML conversion is optional.

[Poly Haven assets are CC0](https://polyhaven.com/license). Their paired diffuse, height, roughness and normal maps in the same UV space can support a replacement backend. Separate rendered lighting, exposure and camera augmentations can bridge diffuse-only training inputs to lit phone photographs without rewriting numeric targets. The small automatic check set measures fitting on familiar sources; fresh materials and real photos judge generalization. Preserve UV scale/height conventions and original assets. Acquire assets through permitted downloads/API use; CC0 does not authorize scraping the website. Dataset preparation remains available, but this release starts no new training run.

## Local decision adviser

[Ollama v0.40.0 release notes](https://github.com/ollama/ollama/releases/tag/v0.40.0) establish MLX decision support, including Clef/Clef Flash, and automatic MLX execution for supported architectures on Apple Silicon. These release notes supersede older System One documentation excluding MLX. The app requires 0.40.0 or newer and uses exactly [clef:27b-nvfp4](https://ollama.com/library/clef:27b-nvfp4).

The adviser calls [System One](https://docs.ollama.com/capabilities/decision) with fixed typed choice questions and one bounded image. Returned labels and probabilities are validated. A rationale is assembled from these choices; outputs are never scripts or executable instructions. The user's **Apply** action maps choices to conservative settings and renders a new preview. Fixed choices reduce variation but are not a guarantee of bitwise deterministic judgments.

Requests stay on loopback, use `keep_alive: 0`, and pass a unified-memory check. Pulling the roughly 18 GB model and deleting it from Ollama's shared library are clearly identified UI actions. Successful document/API compatibility is distinct from a locally verified inference and benchmark; only actual runtime evidence justifies a speed claim.

## Remaining production quality work

- Compare real surfaces under neutral/grazing Cycles lighting, with repeat tiling and actual spatial captures.
- Add optional seam treatment that preserves unique photo patterns.
- Validate a replacement material refiner and restoration model on real photos and fresh textures, following [model vetting](material-model-vetting.md).
- Benchmark the local MLX adviser's recommendation quality and latency on real material surfaces; local execution and unloading are verified in the [validation record](texture-studio-validation-2026-10-07.md).
- Sign and notarize before a general public distribution.

Current files are usable with their stated formats; these remaining items concern visual quality and distribution confidence.
