# Numerical precision and output resolution

## Preservation is distinct from estimation

Original extraction preserves the decoded sample values, dtype and coordinate system. It does not apply gamma correction, tone mapping, normalization, sharpening or display scaling. A stored RGB code may still have an encoded transfer function: preserving it does not make it linear-light RGB. A numeric depth/disparity plane is treated as data.

A camera or encoder may already have discarded information. Float32 storage cannot restore missing levels in an 8-bit encoded map, and an Apple float16 decoded representation does not establish 16-bit scene measurement precision. A teacher estimate or interpolated map is a new derived product; it is not the original auxiliary array or a measured reference.

**Meters**, **inverse meters**, **pixel disparity**, **signed flow** and **normalized displacement** have different meanings. Depth in meters grows with distance. Disparity commonly grows as objects approach the cameras; its sign and principal-point correction depend on the defined coordinate convention. Use the supplied calibration and product metadata rather than guessing from whether a preview looks bright or dark. Normalized displacement is an explicit effects conversion; it must not replace raw or calibrated data.

Invalid samples can be `NaN`. Preserve them and decide how your destination renderer handles unsupported geometry. Replacing invalid values with zero can produce false troughs or infinite-distance surfaces. A dense RAFT estimate retains uncertain predictions; the separately named supported-depth output applies the stricter correspondence mask. Dense output and independently supported geometry are different products.

## Native stereo versus the display image

A Spatial Photo may contain a **2688 × 2016 left/right pair** and a separate **5712 × 4284 display image**. These are distinct camera grids. The output sizes are examples from these captures, not requirements imposed on every file.

Dataset teachers receive the **full display RGB photo**, not either stereo view. Relative-teacher metric anchors receive that same display photo. The original display-sized predictions and the model's native predictions are retained separately. A teacher output on the display grid therefore stays useful for direct inspection or effects work, even when it cannot supply a stock RAFT training target.

Stock RAFT-Stereo predicts correspondence on its left input grid. Its existing decoder does not directly produce an independent 5712 × 4284 prediction from 2688 × 2016 inputs. Training uses crops taken from native left/right arrays and aligned native-grid targets, rather than shrinking the whole stereo pair into a small training image. A 512-pixel crop contains 512 original stereo pixels across. Larger crops increase context and memory cost. More recurrent iterations allow more refinement, also at increased cost. Neither setting repairs incorrect or blurry labels.

The display-teacher workflow does **not** automatically register its targets into the left stereo grid. The display photo can have a different viewpoint, making alignment depend on scene depth. A single image warp or matching aspect ratio does not solve that geometry. The experimental stereo-to-display student instead learns directly against the untouched display target and predicts each requested display pixel from native stereo inputs. There is no fallback that reruns the teacher on stereo-left RGB. Stock RAFT remains a separate native-left baseline.

Upscaling stereo inputs or targets to 5712 × 4284 creates more sample positions, not new measured stereo detail. Existing stereo/display registration diagnostics are approximate derived products with their own validity limits; they are not the default display-teacher training path. Keep them separate from a teacher's direct display-grid estimate. Failed registration must not become an implicit label alignment. Full-sized output is useful for a displacement texture but is not proof of full-sized geometric information.

Keep three teacher sizes separate: the **source RGB dimensions** of the full display photo; the **effective model input dimensions** after that teacher's required processing; and the **stored source-grid output dimensions**. An output saved at 5712 × 4284 can still contain a smooth estimate if the model processed a smaller grid. Raising configurable model input size costs more memory/time; assess the resulting detail and accuracy on reviewed examples.

DepthPro internally resizes its full reference to **1536 × 1536** and uses a 1536/768/384 pyramid of 384-pixel patches. Its existing checkpoint/decoder has a fixed geometry; increasing this app's configurable-teacher size cannot make DepthPro process native 5712 × 4284 pixels. Its source-grid map is reconstructed from the native prediction using the documented inverse-depth resize.

Configurable Depth Anything V2/DA3 teachers default to native input (zero) in the GUI; the field also applies to a secondary configurable teacher. Zero requests native input: V2 uses the source's shortest side and DA3 its longest side, rounded to their required 14-pixel model grid. With DA3 and an exact 5712 × 4284 reference, both dimensions are already divisible by 14, so native preprocessing preserves that spatial size. This does not guarantee that the model fits available memory. Inspection metadata records requested and actual model sizes. Existing targets are not made sharper by changing a setting for future generation. See the [DepthPro inference implementation](https://github.com/apple/ml-depth-pro/blob/9e65e4dbe9568d23c546fcec53302b10445e109e/src/depth_pro/depth_pro.py) and [DA3 input processor](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/io/input_processor.py).

## Choose an export format

| Format | Appropriate data | Precision and tradeoffs |
| --- | --- | --- |
| OpenEXR | Floating depth, disparity, height and other numerical planes. | A single 32-bit `FLOAT` channel with lossless ZIP compression can store a scalar map with no RGB or alpha. Retain the existing float32 values; avoid HALF conversion and lossy codecs. |
| NPY / lossless NPZ | Exact scientific arrays, including integer dtypes and arbitrary shapes. | Retains dtype, shape, byte order and sample bits. NPZ is a compressed container, not an image normalization. |
| PNG | Unsigned 8-bit or 16-bit supported image planes. | Lossless integer exchange; cannot faithfully store arbitrary floating depth. |
| TIFF | Supported integer/float rasters and image exchange. | Select a lossless numerical export; receiving programs may still interpret it as a display image. |
| JPEG / screen captures | Viewing only. | Unsuitable for precision-preserved depth or training labels. |

A 5712 × 4284 single-channel float32 array contains 97,880,832 bytes (about 93.3 MiB) before compression. The ZIP result depends on the values; there is no guaranteed compression ratio. Float32 is adequate to preserve an existing float32 result, but it does not establish absolute scene accuracy. For float16 sources, retain their precision rather than inventing new precision. For integer raw source codes, retain an exact integer companion instead of silently casting everything to float32.

IPDE verifies numerical exports by reading them back. EXR is an exchange file; a dataset also needs camera calibration, validity masks, source identities, teacher provenance and split assignments. One EXR can replace a scalar depth payload, not the entire dataset's required information.

The [OpenEXR technical introduction](https://openexr.com/en/latest/TechnicalIntroduction.html) documents sample types and compression. The [NumPy save documentation](https://numpy.org/doc/stable/reference/generated/numpy.save.html) describes NPY array storage.
