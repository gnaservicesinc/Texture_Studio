# Spatial alignment and training checkpoints, 2026-10-04

## IMG_1689 display mismatch

The original `/Users/andrewsmith/Downloads/IMG_1689.HEIC` has a 5712×4284
display image and 2688×2016 stereo images. Scaling the display to stereo
dimensions does not establish camera registration. Apple metadata identifies
the display camera as Right (image index 1); the canonical stereo left is image
index 2. The supplied PNG named `Display_vs_left` matches a difference against
the canonical right image, and `Display_vs_right` matches canonical left.
The fit still fails for both views, regardless of their filenames.

On held-out feature matches, display-to-right registration has median error
2.03 pixels and 90th percentile error 14.29 pixels; display-to-left has median
4.75 and 90th percentile 29.21 pixels. Only 2/16 right-grid cells and 0/16 left
cells pass local residual checks. Both registrations are rejected. The capture
also carries Apple's ObjectsTooClose and LowImageQuality warnings. These flags
are diagnostic evidence, not an independent depth measurement.

New datasets retain direct native-left teacher labels as their default. Optional
display teachers remain separate diagnostic products. Display-derived maps
cannot be resized and substituted for left-grid depth: different camera
viewpoints require geometric reprojection, and a rejected registration cannot
provide training labels. Pixel masks limit support to validated feature coverage
and reject every invalid value contributing to interpolation. Duplicate feature
positions cannot inflate the held-out registration evidence. Existing datasets are
not rewritten; their selected training target remains visible in review.

Evidence is saved under `out/alignment-1689`. The original HEIC, supplied PNGs
and existing user datasets were read without modification.

## Choosing a training base

The original [RAFT-Stereo project](https://github.com/princeton-vl/RAFT-Stereo)
still recommends its Middlebury checkpoint for in-the-wild inference. Its
published Middlebury fine-tuning recipe starts from the SceneFlow checkpoint.
These are different uses: a useful inference baseline is not automatically the
best initialization for a new training domain.

For this small iPhone experiment, keep Middlebury as a warm-start baseline and
compare a separate SceneFlow-initialized run on exactly the same held-out scene
groups and settings. Both checkpoints are already local. Choose from full-grid
stereo checks and independently measured references as well as teacher-flow
agreement. A lower distillation score establishes agreement with the teacher,
not that the resulting depth is mathematically exact. No checkpoint was changed
or selected automatically. The same upstream repository also provides the
mixed-dataset `iRaftStereo_RVC` checkpoint, already available locally as
`iraftstereo_rvc.pth`. Its instance-normalized RAFT configuration is supported
by the existing loader; it is another compatible comparison candidate rather
than a proven improvement on these iPhone captures.

Newer models worth a separate evaluation include:

- [WAFT-Stereo (2026)](https://github.com/princeton-vl/WAFT-Stereo): the authors
  recommend their SynLarge checkpoints for downstream generalization. The
  published environment and training commands use xformers and GPU-oriented
  infrastructure; MPS compatibility has not been established here.
- [FoundationStereo (2025)](https://github.com/NVlabs/FoundationStereo): a newer
  stereo foundation architecture with a published zero-shot evaluation. The
  official setup is tested on NVIDIA GPUs and uses flash-attn.
- [Fast-FoundationStereo (2026)](https://github.com/NVlabs/Fast-FoundationStereo):
  a speed-oriented family whose official setup uses CUDA PyTorch and xformers.
  Its authors recommend the original FoundationStereo for offline accuracy.

These weights have different architectures and cannot be loaded into the
current RAFT-Stereo trainer. Their published benchmark results do not establish
iPhone close-up performance or M2 speed. An adapter and an MPS operator audit
would be required before adopting one. The current work keeps the verified RAFT
MPS path and makes checkpoint selection explicit.
