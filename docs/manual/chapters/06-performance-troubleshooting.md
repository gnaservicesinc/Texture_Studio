# Performance, troubleshooting and support

## Use the available hardware

Automatic device selection prefers CUDA where available, then Apple MPS, then CPU. On Apple Silicon, choose **MPS** when supported by the installed runtime. MPS is PyTorch's Metal backend; it is separate from Qt drawing. GPU cores do not map to a setting that should be set to the core count. See the [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

Project **File processing threads** controls CPU import, hashing and preparation; automatic uses available cores subject to memory bounds. It is separate from GPU inference/training. Display training encodes the native stereo pair whole. Adjust decoder tiles and iterations according to measured memory and throughput; stock training uses native crops. Accumulation can provide a larger effective update without holding several images in memory simultaneously. A bounded cache reduces repeated array decompression without requiring the whole dataset in RAM.

Thirty GPU cores, twelve CPU cores and 64 GB of unified memory do not guarantee full utilization. Native encoding, decoder tile dispatch, validation, CPU decoding, synchronous device transfers, checkpoint writes and unsupported operations can be bottlenecks. Check the current phase and device before increasing threads. Measure useful image/update throughput and elapsed validation time; high utilization alone is not success. Avoid changing several controls at once.

## Full-display teacher memory

Teacher reference size and model processing size are different controls. Passing the complete display photo preserves its framing and source identity; DepthPro still uses its fixed 1536 × 1536 internals. DA3's native option can process a 5712 × 4284 grid, but native GIANT inference on a 64 GB Mac is **unverified**, not a supported memory guarantee.

At that size, DA3 has 124,848 spatial tokens. The installed GIANT checkpoint contains about 5.05 GiB of FP32 tensors. Its decoder can create a single full-resolution 128-channel FP32 feature of about 11.67 GiB, plus positional embeddings and other large temporaries. These are source-derived tensor sizes, not a measured peak. Upstream decoder/frame chunking does not spatially tile a single photo. Whole-image attention also requires substantial computation. See the [DA3 decoder](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/dualdpt.py).

PyTorch 2.14's MPS attention implementation has an FP32 tiled path compatible with DA3 GIANT's 64-dimensional heads. This can avoid allocating a complete quadratic attention matrix; it does not eliminate the decoder's large tensors or guarantee sufficient memory. Runtime version and selected backend matter. See the [PyTorch MPS attention source](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/Attention.mm).

Run teacher inference serially and use bounded file preparation. Preserve FP32 and inspect the actual model-input dimensions in metadata. If native inference runs out of memory, select an explicit smaller processing size or another teacher and compare the resulting direct display maps; do not assume CPU threads or lossless compression solve accelerator working-memory limits. Generating a display-sized map after a smaller model pass is disclosed resampling, not native-resolution inference. Spatially tiling model crops would change the global prediction and requires its own scale/seam validation; it is not an automatic native substitute.

| Symptom | Investigate | Recommended action |
| --- | --- | --- |
| Blurry full-sized output | Teacher processing size, native predictions, model architecture and labels. | Inspect source/target at native pixels; increase appropriate teacher detail and review before retraining. |
| Depth tile side length | 768 means at most 768 × 768 output pixels in each temporary decoder tile. | 48 portions cover one 5712 × 4284 map as 8 columns × 6 rows; output dimensions stay unchanged. |
| Dataset checking dominates startup | Dataset origin/schema and external import. | Supported native datasets check payloads as consumed; external/unknown datasets require full preflight. |
| Training is on CPU | Reported device and available PyTorch backend. | Select MPS on compatible Apple Silicon; inspect the failure if it is unavailable. |
| Low GPU load between updates | Decompression, preprocessing, synchronization or frequent validation. | Inspect phase timings, use suitable cache/worker settings and reduce validation overhead. |
| Validation error varies | Different random subsets or scarce independent scenes. | Increase sample count or use full validation; compare identical policies. |
| Validation stays good but new scenes fail | Scene leakage or insufficient diversity. | Rebuild an independent split and add genuinely new scenes. |
| Error diverges / zero / NaN | Incorrect calibration, invalid targets, loss or runtime problem. | Read the stop reason; review data before resuming the saved state. |
| Display target is ineligible for stock RAFT | Display and native-left grids represent different outputs/viewpoints. | Use the display depth model for that output task, or compare direct display AI; do not substitute a left teacher or force a registration/resize. |
| EXR looks black in an image viewer | Numerical units, range or NaN handling. | Inspect the values in a data-aware program; use a separate visualization. |
| HEIC rewrite refused | Writer changes retained data or loses auxiliary/calibration items. | Export separate exact arrays; retain the original container. |
| A linked dataset disappears | Its original directory moved or became unavailable. | Restore the source or update the link in Dataset Studio. |

## Install and report issues

macOS releases are pre-releases until 1.0.0 and are not notarized. The finished IPDE Studio bundle contains its project apps and deployed Qt libraries. Source developers can run `make build` and `make install`; installation requires the existing app and its subapps to be closed. `make DESTDIR=/path/to/staging/dir install` stages the application under that root's Applications folder. Keep project data outside the application bundle.

Use **Help → Report Bug** to open the [issue form](https://github.com/gnaservicesinc/ipde/issues/new). Include copied About information, app role, input format/grid sizes, selected products or training settings, steps to reproduce, and the relevant error/report. Share a minimal example when practical; a model can require substantial runtime resources to reproduce. Include whether a dataset is local, linked or imported and whether validation used all images or a subset.

The [IPDE repository](https://github.com/gnaservicesinc/ipde) contains source, releases and development documentation. [pillow-heif documentation](https://pillow-heif.readthedocs.io/en/latest/) covers HEIF decoding. This manual's recommendations are starting points for experiments, not claims of measured accuracy.
