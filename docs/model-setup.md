# Local high-detail depth models

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
or choose **Depth method** in Extractor or use RAFT Studio's model controls. Extractor defaults to direct DepthPro full-display depth; V2 Large and DA3 are selectable without using a trained RAFT checkpoint. Its normal CLI accepts `--learned-depth --learned-model depthpro`, `--learned-model-path`, `--learned-source-dir`, `--learned-device`, and `--learned-input-size`. Select `learned-display-depth` for full-photo float32 EXR, `learned-display-native` for the model grid, and `learned-display-preview` for the separate PNG. For example:

```sh
PYTHONPATH=src .venv/bin/python -m ipde --learned-depth --learned-model depthpro \
  --select learned-display-depth --select learned-display-native --manifest \
  --output-dir /path/to/new/exports /path/to/photo.HEIC
```

Device choices are `auto`, `mps`, `cuda`, and
`cpu`; `auto` prefers CUDA, then Apple Metal, then CPU. A missing explicitly
selected file produces an actionable error; another checkpoint is never silently
substituted.

DepthPro uses a fixed 1536×1536 internal prediction. For Depth Anything V2,
`input_size=1036` increases the processing grid from the upstream 518 default;
it preserves aspect ratio and rounds to multiples of 14. This costs more memory
and compute and does not guarantee more accurate geometry. DA3 uses this option
as the longest-side processing bound, while V2 uses the shortest-side bound.
RAFT Studio's GUI and CLI default to 1036. The GUI migrates an older saved zero
to 1036 once when DA3 is selected, including as an additional teacher or for
per-photo generation. An explicitly chosen custom size remains available.
Zero requests native processing; DA3 rejects a resulting grid above 8192
14×14 patches before loading weights or running inference. Use 1036 or smaller
if the grid is rejected. Original RGB and full-size float depth results remain
retained, but interpolating model output to full photo dimensions does not add
independently predicted detail.
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

`native_depth` retains the network prediction grid, and `source_depth` is a
separately identified source-grid derivative. DepthPro source-grid conversion
interpolates inverse depth before applying Apple's reciprocal guard, matching
its inference API. Interpolating metric depth would give different values.
Upsampling to a 5712×4284 photo adds samples but does not create independent
measured detail. No prediction is normalized, gamma corrected, tone mapped,
quantized to 8-bit, sharpened, or blended into the extracted native depth.
PNG previews and normalized displacement products are separate requested
derivatives. Use the float32 NPY/EXR products for numerical work.

Use the calibrated spatial-left focal length only with the spatial-left RGB
grid. The higher-resolution display image has a different camera/frame; when
its own calibration is unavailable, DepthPro estimates its focal length.
Never borrow the left stereo focal value for the display image.

These are estimated depth maps, including the metric model. They are useful
teachers or displacement starting points, but their visual sharpness is not
proof of accurate geometry. Teacher predictions alone are pseudo-labels;
real calibration, withheld scenes, and independent depth measurements are
needed to establish training gains.

On the supplied IMG_1148 and IMG_1168 HEICs, all three models completed actual
FP32 Metal inference for both the 2688×2016 spatial-left image and the 5712×4284
display image. After model loading, the six spatial/display combinations took
roughly 2.2–3.5 seconds per image on the available M2 Max. DA3 at a 1036 longest
side predicted a native 1036×784 grid; V2 at a 1036 shortest side predicted
1386×1036. The DA3 process reported a 10.47GiB peak resident size and sampled
10.12GiB Metal driver allocation after inference; the latter is not a peak
measurement. These are runtime/numerical checks, not measured geometry scores.

Model licenses differ. Consult [Apple DepthPro](https://huggingface.co/apple/DepthPro),
[V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large),
[V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small), and
[DA3 Giant1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) before using
or redistributing model-derived training outputs/checkpoints.
