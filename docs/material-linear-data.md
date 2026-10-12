# Diffuse color and numeric material maps

New native compact RGB models use linear-light sRGB diffuse tensors. A source
marked sRGB is decoded once at tensor creation; a source marked linear retains
its normalized channel values. This does not change the source image or any
prepared crop. Tensor conversion is separate from pixel geometry and storage.

Checkpoints record `input_tensor_transfer = linear-srgb-v1`. Training,
continuation, baseline comparison, and inference all read that contract. Older
checkpoints without the field retain their original sRGB code input convention,
as does the pretrained PBRnxt family. Continuing an older run does not silently
change its input distribution; start a fresh compact run for the linear-light
input contract.

Height, roughness, and encoded normal channels are numeric data. The native
loader divides integer codes by their code maximum, with no display gamma,
contrast normalization, or denoising. PNG color metadata does not alter those
numeric samples. The model uses a linear output head and exports Float32 EXR
values unchanged. In Blender, set these textures to **Non-Color**. A display
profile or a grayscale appearance does not establish that height samples are
gamma encoded; blindly applying an sRGB transfer would change the learned
geometry and turn a neutral normal component of 0.5 into about 0.214.

[Blender's color-management manual](https://docs.blender.org/manual/en/3.3/render/color_management.html)
specifies Non-Color for normal and displacement data so no color-space
conversion is applied.
[Apple's Core Image working-color-space documentation](https://developer.apple.com/documentation/coreimage/cicontextoption/workingcolorspace)
describes its default linear working space and `NSNull` for numeric processing
without color management.

## Supplied 1K contrast check, 2026-10-11

These are measurements of the supplied files under
`/opt/ipde/testing/Texture Studio Maps ECFE0080/`, not a claim about their training
order or an independent model-quality benchmark. Values were read directly from
the 16-bit grayscale PNG codes and uncompressed Float32 EXR scanlines.

| File | Minimum | Maximum | Mean | Standard deviation | 1st–99th percentile |
| --- | ---: | ---: | ---: | ---: | --- |
| `anti_skid_tiles_1k_full-Source-displacement-reference.png` | 0.449561 | 0.962020 | 0.703932 | 0.0514965 | 0.582223–0.839719 |
| `anti_skid_tiles_1k_full-export-Displacement-63f7400691dd.exr` | 0.539741 | 0.553454 | 0.545845 | 0.000611749 | 0.543768–0.547038 |

Both are 1024 × 1024 scalar maps. The EXR contains no nonfinite samples. Its
standard deviation is about 84 times smaller than the reference, confirming
that the low contrast is present in the numeric data. Display color management
alone does not explain the flattening.

SHA-256 identities:

- Reference PNG: `93414169c300568a2c6c1e01bbc6c7ab86e0bfdf4a54a4d45201c77d5b9851d9`
- Export EXR: `21ca75b625949e89d6166292a8c03596399b7591903ab25206f36e0d44958f0e`

The compact model's loss remains value L1 plus four-scale gradient L1; there is
no sigmoid output or post-inference contrast squeeze. Its initial scalar bias
is 0.5, but no operation forces later predictions to 0.5. Broad or ambiguous
diffuse-to-height supervision can favor low-detail predictions under L1; that
is a possible explanation, not a demonstrated diagnosis of this run. Focusing
training by material type is worth evaluating with held-out source materials.
The linear input correction does not establish that contrast collapse or
material quality is fixed.
