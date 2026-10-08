# Optional Python depth models

This page describes the retained Python extraction and research commands.
Texture Studio manages DA3-GIANT-1.1 and a separate PyTorch/MPS runtime in its
**Local Models** window. Its bundled worker is independent of the older research
source folders and commands below. Optional custom Core ML models are also
supported. Apple Small V2 and embedded portrait depth are not material sources.
You can attach a registered exported map where its licensing permits use.

Run the setup script explicitly to install the optional dependencies and download
revision-pinned model sources/checkpoints. Inference and importing IPDE never
download files or send photos anywhere.

```sh
bash scripts/setup-depth-models.sh depthpro depth-anything-v2
# Optional comparison baselines:
bash scripts/setup-depth-models.sh depth-anything-v2-small depth-anything-3
```

The default prefix is the repository's parent (`/opt/ipde` here), and the default
Python is its `.venv/bin/python`. Both accept overrides via `--prefix` and
`--python`. Existing edited model-source checkouts are preserved. The adapter
loads upstream source directly; upstream packaging pins that would downgrade
IPDE's NumPy or replace the installed Torch are not applied.

| Choice | Local checkpoint | Source directory | Numerical output |
| --- | --- | --- | --- |
| `depthpro` | `/opt/ipde/models/depth_pro.pt` | `/opt/ipde/ml-depth-pro` | Estimated depth in meters, with calibrated or estimated focal scaling |
| `depth-anything-v2` | `/opt/ipde/models/depth_anything_v2_vitl.pth` | `/opt/ipde/Depth-Anything-V2` | Relative inverse depth; larger values are nearer |
| `depth-anything-v2-small` | `/opt/ipde/models/depth_anything_v2_vits.pth` | `/opt/ipde/Depth-Anything-V2` | Relative inverse depth; larger values are nearer |
| `depth-anything-3` | `/opt/ipde/models/DA3-GIANT-1.1/` | `/opt/ipde/Depth-Anything-3` | Relative depth; smaller values are nearer |

Configure `LearnedDepthConfig(model, model_path, source_dir, device, input_size)`
or use the extraction CLI options. Its built-in depth sources are **DepthPro**,
**DA3** and **DA2**. Each saves one raw float32 EXR at the full display-photo
dimensions. The CLI source IDs are `learned-depthpro`,
`learned-da3` and `learned-da2`; `--learned-depth` alone exports the selected
`--learned-model` (DepthPro by default). For example:

```sh
PYTHONPATH=src .venv/bin/python -m ipde --learned-depth --no-npy \
  --select learned-depthpro --select learned-da3 --select learned-da2 \
  --output-dir /path/to/new/exports /path/to/photo.HEIC
```

Device choices are `auto`, `mps`, `cuda`, and
`cpu`; `auto` prefers CUDA, then Apple Metal, then CPU. A missing explicitly
selected file produces an actionable error; another checkpoint is never silently
substituted.

DepthPro uses its fixed 1536×1536 internal model grid. DA2 and DA3 default to
`input_size=0`, processing the full display photo with the model's required
14-pixel grid rounding. An explicit nonzero size reduces input resolution:
DA2 uses a shortest-side bound and DA3 uses a longest-side bound. The saved
depth map always matches the display-photo dimensions.
DA3's model directory must include its `config.json` and `model.safetensors`.
IPDE constructs exactly the network serialized in that config and bypasses the
public API's forced mixed precision. Its unrelated web/export/CUDA packages are
not needed. Upstream declares Python≤3.13 in its package metadata; the direct
source adapter is validated separately in IPDE's Python3.14 environment.

All model parameters and arithmetic are FP32 during inference, without
autocast; CUDA TF32 is explicitly disabled during the inference call. The
original Apple DepthPro checkpoint itself stores FP16 weights;
promotion to FP32 cannot recover the missing weight bits. V2 Large stores FP32
weights. Each result records the actual checkpoint dtypes and SHA-256, source
revision, input hash, device, processing, units, and resampling method.

The extraction source is the model's `source_depth`, stored unchanged as
float32 EXR; exact NPY companions are optional. DepthPro's required source-grid
conversion follows Apple's inverse-depth resize. No prediction is normalized,
gamma corrected, tone mapped, quantized, sharpened or blended with Apple depth.
Native prediction grids, preview PNGs and normalized AI displacement maps
are not separate extraction sources.

Use the calibrated spatial-left focal length only with the spatial-left RGB
grid. The higher-resolution display image has a different camera/frame; when
its own calibration is unavailable, DepthPro estimates its focal length.
Never borrow the left stereo focal value for the display image.

These are estimated depth maps, including the metric model. They are useful
teachers or displacement starting points, but their visual sharpness is not
proof of accurate geometry. Teacher predictions alone are pseudo-labels;
real calibration, withheld scenes, and independent depth measurements are
needed to establish training gains.

In an earlier runtime check on the supplied IMG_1148 and IMG_1168 HEICs, all three models completed actual
FP32 Metal inference for both the 2688×2016 spatial-left image and the 5712×4284
display image. After model loading, the six spatial/display combinations took
roughly 2.2–3.5 seconds per image on the available M2 Max. DA3 at a 1036 longest
side predicted a native 1036×784 grid; V2 at a 1036 shortest side predicted
1386×1036. The DA3 process reported a 10.47GiB peak resident size and sampled
10.12GiB Metal driver allocation after inference; the latter is not a peak
measurement. These reduced-size timings are historical and do not describe the
current native-input default. These are runtime/numerical checks, not measured geometry scores.

Model licenses differ. Consult [Apple DepthPro](https://huggingface.co/apple/DepthPro),
[V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large),
[V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small), and
[DA3 Giant1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) before using
or redistributing model-derived training outputs/checkpoints.

The retained dataset CLI generates DepthPro by default. An earlier full-photo
DA3 check took about 20 minutes for a 5712 × 4284 image; DA2 also took minutes.
These historical timings explain why Texture Studio bounds model input
independently of the final texture size. DA3 uses PyTorch/MPS in the native app.
Existing dataset maps remain usable
with the retained Python tools.
