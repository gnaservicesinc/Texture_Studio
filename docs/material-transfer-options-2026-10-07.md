# Pretrained material-detail options — 7 October 2026

This source tree contains the implementation and documentation. Experiment checkpoints, generated maps, source snapshots, recovery recipes and reports are retained locally and are not included. Record names below identify that local archive.

Numeric fitting error is one diagnostic, not the definition of a better material. Use the later [displacement review workflow](material-quality-workflow.md) to assess useful relief, misplaced bumps/pits, noise, and overall realism. The historic values and model identities below remain unchanged.

The first **DINOv2 Base features plus a native-resolution material decoder** comparison is complete: it slightly improved some detail metrics on the repeated training crop, but did not improve the disjoint region of that material. DeepBump also has a measured pretrained material-normal baseline. DA3 attention adapters remain a proposed comparison. These results do not establish that a larger model or LoRA improves this dataset.

The target is relative surface relief from photographed albedo. Scene distance and camera geometry are useful priors, but neither is a displacement reference. Keep the original UInt16 PNG targets; convert their numeric codes to Float32 only when loading batches. Keep a native RGB path to the height decoder so pores, fibers and grains remain visible even if the pretrained contextual branch processes a smaller image.

## Verified candidates

Sizes and tensor counts below come from the official Hugging Face model metadata fetched on this date. They describe downloaded weights, not peak training memory. Some model cards round parameter counts differently. All listed safetensors are Float32. The metadata snapshots are in `out/material-training/transfer-model-metadata/`.

| Candidate | Actual tensors | Weight bytes | License / access | Intended experiment |
| --- | ---: | ---: | --- | --- |
| [DINOv2 Small](https://huggingface.co/facebook/dinov2-small) | 22,056,576 | 88,249,960 | Apache-2.0, open | Integration smoke and lower-cost feature baseline |
| [DINOv2 Base](https://huggingface.co/facebook/dinov2-base) | 86,580,480 | 346,345,912 | Apache-2.0, open | First larger pretrained feature encoder; frozen initially |
| [DA3 Small](https://huggingface.co/depth-anything/DA3-SMALL) | 34,299,463 | 137,248,940 | Apache-2.0, open | Adapter and gradient plumbing smoke |
| [DA3 Base](https://huggingface.co/depth-anything/DA3-BASE) | 135,366,599 | 541,518,028 | Apache-2.0, open | First scene-depth transfer candidate |
| [DA3 Large 1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | 410,941,767 | 1,643,843,860 | Apache-2.0, open | Larger comparison after measured Base training feasibility |
| [DINOv3 Small](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 21,596,544 | 86,406,384 | Custom DINOv3 license, manual access gate | Later dense-feature comparison |
| [DINOv3 Base](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 85,660,416 | 342,662,192 | Custom DINOv3 license, manual access gate | Later comparison; newer does not prove better material relief |

DINOv2 provides dense patch features and intermediate layers, rather than a material-height output. Its patch size is 14, and the verified Base configuration has 12 layers and 768 feature channels. Train a separate height decoder against the actual high-bit-depth displacement crops. The official implementation exposes `get_intermediate_layers`; the Transformers implementation exposes hidden states. Use image normalization expected by the pretrained encoder only on the encoder input, with no color conversion on the displacement or normal references. [Official implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py), [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/dinov2).

DINOv3 is not under the Apache license used by DINOv2. Its license covers the pretrained models and fine-tuning material and imposes its own redistribution terms. The public model metadata was readable, but unauthenticated configuration download returned HTTP 401. Prefer the simpler openly downloadable Apache candidates for the first implementation. [Official DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

## LoRA integration that actually trains

LoRA applies to vision models and custom PyTorch modules. Identify the exact linear layers and separately save the newly trained material head; do not depend on a language-model default target list. Start with rank 8 or 16 as an experiment, report the real trainable parameter count, then compare frozen encoder, adapters and partial unfreezing under the same supervision. [PEFT custom model documentation](https://huggingface.co/docs/peft/main/en/developer_guides/custom_models).

The checked DA3 source revision is `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. `DepthAnything3.forward` is decorated with `torch.inference_mode()` and also enters `torch.no_grad()`. **Do not train adapters through that public API.** Use the underlying differentiable network or its backbone directly; require finite nonzero adapter and decoder gradients before starting a run. The backbone contains fused attention `qkv` and output `proj` linear layers. Constrain target names to attention blocks, since patch embedding also has a `proj` module. The Base backbone returns layers 5, 7, 9 and 11; Large returns 11, 15, 19 and 23. [Pinned API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/api.py), [pinned attention implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/dinov2/layers/attention.py).

DA3's shipped depth head estimates scene geometry, with additional camera branches and postprocessing. A material decoder should supervise surface relief directly rather than turning that scene depth into a supposed height target. Calling the underlying backbone also avoids training unused camera and scene postprocessing branches. This is an implementation choice inferred from the source architecture, not an upstream material-training recipe. [Pinned network implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/da3.py).

64 GB makes these medium-sized candidates worth measuring. LoRA reduces optimizer and gradient storage for frozen weights; it does not eliminate transformer activations or native-resolution decoder activations. A full Float32 Adam update of all 410.94M Large tensors alone accounts for about 6.58 GB for weights, gradients and two moment arrays, before activations and temporary storage. Thus weight size is not a training-memory estimate. Keep native 1024/2048 targets and a native detail branch; initially process the contextual encoder at 504/518 pixels or use explicitly overlapped native tiles. Do not silently downsample the entire supervised job. Benchmark one forward/backward/optimizer step on MPS at batch one, record memory and wall time, and test the real gradient path. [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

## DeepBump baseline

[DeepBump](https://github.com/HugoTini/DeepBump) is GPL-3.0 and directly predicts material normals from image tiles. The official repository says its training code is unavailable. It is therefore a useful pretrained baseline, not a ready-made LoRA training project.

Pinned revision: `fad19ba87daed12b1d0410a57e74f3d79e82f78d`. `deepbump256.onnx` is **26,706,979 bytes**, Git blob identity `01158e2ca4800c3365b2b5b539f7118cc9c5da60`. The optional upscale model is a separate 17,263,442-byte artifact and is unnecessary for this comparison. [Pinned model](https://github.com/HugoTini/DeepBump/blob/fad19ba87daed12b1d0410a57e74f3d79e82f78d/deepbump256.onnx), [pinned license](https://github.com/HugoTini/DeepBump/blob/fad19ba87daed12b1d0410a57e74f3d79e82f78d/LICENSE).

`module_color_to_normals.apply` takes CHW float RGB, computes a grayscale channel mean, predicts native 256-pixel tiles, blends overlaps, then normalizes vectors. The upstream implementation explicitly selects ONNX Runtime's CPU provider. The stock CLI divides image input by 255 and exports UInt8, so bypass that CLI when evaluating high-bit-depth input or saving numeric results. [Pinned normal API](https://github.com/HugoTini/DeepBump/blob/fad19ba87daed12b1d0410a57e74f3d79e82f78d/module_color_to_normals.py), [pinned CLI](https://github.com/HugoTini/DeepBump/blob/fad19ba87daed12b1d0410a57e74f3d79e82f78d/cli.py).

Its normals-to-height step is Frankot–Chellappa gradient integration, not another learned model. It approximates gradients with X/Y normal components rather than X/Z and Y/Z ratios, then stretches each result to its own minimum and maximum. Preserve a float normal baseline and label any height integration as relative, with unknown offset and scale; do not claim upstream min/max output retains calibrated height. Verify the normal axis convention against the OpenGL reference and report any alternative-axis comparison separately rather than silently selecting the better score. [Pinned height implementation](https://github.com/HugoTini/DeepBump/blob/fad19ba87daed12b1d0410a57e74f3d79e82f78d/module_normals_to_height.py).

## Comparison and release criterion

Use the same disjoint native validation regions for each candidate. Report this as **unseen regions of known materials**; it does not measure generalization to fresh materials or camera photographs. Show height error without per-image contrast stretching, offset-invariant height error, native and multiscale gradient error, high-frequency error, and OpenGL normal angular error. Preserve raw Float32 predictions alongside display previews. Paired normal and displacement maps can have different artistic strength, so direct reference-normal error and normals derived from the displacement answer different questions.

Choose the model only after an observed improvement in native detail and usable Blender output. Bigger contextual features, a successful backward pass and a decreasing training loss do not by themselves establish better displacement.

## Published 2K source maps, separate from the active dataset

`scripts/download_material_resolution.py` plans or downloads the site's own original 2K PNG map sets. It never resizes the existing 4K parents or converts their gamma/precision. By default it only plans; `--download` is explicit. Selection comes from verified Poly Haven asset IDs in the current dataset, or explicit official asset IDs. It chooses diffuse, displacement, OpenGL normal and roughness PNGs, without substituting JPEGs. The official API provides per-file sizes and full MD5 checksums, and Poly Haven's asset license is [CC0](https://polyhaven.com/license).

The fresh metadata plan for all **55 verified Poly Haven materials** includes **220 PNGs totaling 3,139,853,183 bytes (2.924 GiB)**. This is an exact published-byte plan, not a download or a claim that all 220 headers have been examined. Only the requested two-material pilot was obtained: **farm_soil and white_stucco_02, eight files totaling 125,341,362 bytes**. Each file matches its fresh published MD5 and byte count, with a recorded full SHA256. Actual IHDR headers show all eight files are UInt16, and both four-map sets are native 2048×2048. The downloader supports rectangular map sets when all four dimensions agree. Published UInt8 normal or roughness maps remain UInt8 with explicit precision notes; displacement requires UInt16 for this high-detail training path. No upconversion invents source precision.

The pilot originals and per-material provenance are stored under `/opt/ipde/material-dataset/sources-2k/farm_soil/` and `/opt/ipde/material-dataset/sources-2k/white_stucco_02/`. A second run verified/reused all eight files with **zero PNG bytes downloaded**, leaving the manifests untouched. The active dataset index was unchanged throughout. Current reports are `out/material-training/published-2k-all-materials-plan-02.json`, `published-2k-pilot-download.json` and `published-2k-pilot-reuse.json`. Twelve CPU tests cover source bit depth, strict official hosts, original-byte identity, rectangle pairing, no-clobber publication, atomic metadata, idempotent reuse, failed temporary-file cleanup and pre-download byte-budget enforcement.

These are another published resolution of the same material, not an independent camera exposure. The manifest retains the existing 4K parent dimensions, source hashes and train/validation rectangles, but **cross-resolution geographic registration remains unverified and no training split is assigned**. Do not add 2K crops to training until their footprints are checked against every existing validation region. A native 1024-pixel crop from the 2K variant can cover roughly twice the width and height of a 1024-pixel crop from the corresponding 4K variant, and can overlap held-out pixels. The user has tested Poly Haven's published-resolution quality; this does not establish the same result for another provider. Wicker013 is outside this Poly Haven download path.

Example bounded pilot command, using a new report filename on each run:

```sh
.venv/bin/python scripts/download_material_resolution.py \
  --dataset /opt/ipde/material-dataset --resolution 2k \
  --materials white_stucco_02 --materials farm_soil --download \
  --destination /opt/ipde/material-dataset/sources-2k \
  --max-bytes 268435456 --report out/material-training/published-2k-pilot-next.json
```

## Measured DeepBump baseline

The wrapper `scripts/evaluate_deepbump_material.py` ran the pinned model on six existing 1024×1024 validation regions, using two CPU threads and 81 overlapping native 256-pixel tiles per image. All six finished in 28.5 seconds. The 18 input/reference PNG full SHA256 hashes and dataset index stayed identical, and each exported Float32 EXR passed exact numeric readback. No upstream UInt8 CLI was used. The model's verified full SHA256 is `3a2ababc5fa652b19040a0f3d639b71826368e52ad805cf70ffb0e7c96a54d67`.

| Validation material | Mean normal error | Flat normal error | Integrated-height shape correlation |
| --- | ---: | ---: | ---: |
| white_stucco_02 | 21.216° | 18.873° | −0.0753 |
| farm_soil | 32.927° | 37.838° | 0.0789 |
| cotton_jersey | 7.856° | 8.195° | −0.0016 |
| broken_brick_wall | 30.140° | 31.665° | 0.1022 |
| wicker013 | 21.891° | 20.773° | 0.0291 |
| wood_planks | 7.811° | 5.297° | 0.2660 |

These results do not establish a useful displacement model. The pretrained normal predictor improves on a flat normal for three of six crops, and its integrated height has weak correspondence to the original displacement. Fixed OpenGL axis interpretation was used; alternate flipped-Y scores remain separate diagnostics. Wicker013's provider was unresolved when this baseline ran; its four original files subsequently matched AmbientCG's official CC0 Surface Photogrammetry package exactly. It is a separate provider, not a Poly Haven asset. Five diffuse inputs and all reference normals are UInt16; Wicker's diffuse is UInt8. Float32 exports preserve computed predictions, not extra captured source precision.

Raw normals, mean-zero uncalibrated integrated heights, source checksums and per-sample metrics are in `out/material-training/deepbump-native-validation-01/`, with the complete report in `comparison.json`. The wrapper has 16 CPU regression tests covering pin verification, native UInt16 preservation, explicit color transfer, alpha exclusion, OpenGL integration signs and amplitude, exact FLOAT EXR export, source dimensions and precision, split scope and budget enforcement. Metric masking excludes undefined reference normals inside ineligible pixels; it does not guarantee that convolutional context contains only valid pixels. Global height integration is skipped for partially masked crops, since invalid-region gradients can otherwise influence the whole solution. The six completed crops were fully eligible and passed the added array contracts with unchanged angular metrics. It does not train DeepBump or silently promote it into the app.

## Measured frozen DINOv2 comparison

`scripts/diagnose_frozen_dino_height.py` completed **300 actual MPS optimizer updates per variant**, with identical initial native-head weights, learning rate 0.001, AdamW settings, source crop and relative-squared objective. The zero-feature control keeps the same projection and fusion architecture as the feature variant. Both initial predictions were asserted to be exactly 0.5. This comparison trains a 328,957-parameter native RGB/height head; all 86,580,480 DINOv2 Base parameters remain frozen. **It is not LoRA.**

The training sample is `white_stucco_02_auto_001`, source rectangle `[0,0,1024,1024]`. The separate sample is `white_stucco_02_auto_003`, rectangle `[1536,1536,1024,1024]`, from the same material with no overlapping source pixels. Both source/target pairs are fully eligible for loss and metrics. Original UInt16 displacement codes are unchanged and become Float32 only in the active batch. The native RGB/detail head remains 1024×1024; only the encoder input is converted to its expected sRGB normalization and resized to 518×518. Its final normalized patch features have 768 channels on a 37×37 grid, with a learned 12-channel projection before spatial upsampling.

| Evaluated pixels | Conditioning | Centered height correlation ↑ | Radius-4 detail correlation ↑ | Native gradient MAE ↓ | Centered height MAE ↓ | Relative-squared sum ↓ | Height amplitude/std ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated training crop | Zero features | 0.989367 | 0.892523 | 0.001862519 | 0.004927939 | 0.335175 | 0.984740 |
| Repeated training crop | Frozen DINOv2 | 0.987994 | 0.903567 | 0.001814801 | 0.005218063 | 0.305720 | 0.967757 |
| Disjoint known-material region | Zero features | 0.871406 | 0.653801 | 0.002732090 | 0.017420257 | 1.244408 | 0.902759 |
| Disjoint known-material region | Frozen DINOv2 | 0.842308 | 0.645463 | 0.002876596 | 0.019000117 | 1.381454 | 0.867006 |

The relative-squared sum adds the centered-height, multiscale-gradient and high-pass MSE terms, each measured in fixed target-energy units; it does not rescale target pixels. Raw height error is also retained in the complete report. Its absolute offset can differ because the training objective intentionally ignores a constant height origin. An amplitude ratio near one alone does not prove correctly located relief.

Frozen features gave a modest training-detail gain: high-pass correlation rose from 0.8925 to 0.9036 and training objective fell from 0.3352 to 0.3057. On the **unseen region of the known material**, height/detail correlations and every component of the relative-squared objective worsened; the sum rose from 1.2444 to 1.3815. There is no observed region-quality win and no reason to select this feature variant as a new default. This is one material, one training crop and one seed. It neither rules out other pretrained features or adapters nor establishes results on all 56 materials or fresh photographs.

Frozen-feature extraction took 1.830 seconds. The 300-update loops took 49.768 seconds for zero features and 49.922 seconds for frozen features, including periodic training metrics. The sampled Metal driver peak was 2,200,502,272 bytes; process peak RSS was 1,151,795,200 bytes. Those measurements can overlap and must not be added. The two feature caches together use 8,411,136 bytes; all output artifacts total 166,555,923 bytes. A separate decoder training-memory measurement is not a measurement of LoRA memory.

The actual step-2 native RGB encoder gradients were nonzero in both variants; the feature projection gradient was nonzero only with frozen features. At step 1, native encoder gradients were zero because the output head starts at zero. Its parameter-change flag is nevertheless true because AdamW also applies weight decay: **a changed parameter at that first step is not evidence of a nonzero learning gradient**. The step-2 measurements supply that evidence. The frozen pretrained encoder was detached before head optimization and never trained.

Complete precision, crop, checkpoint, code-hash, timing and metric evidence is in the local report `out/material-training/frozen-dino-stucco-01/summary.json`. Both final checkpoints and native Float32 EXR/NPY height and OpenGL-normal outputs are retained locally as diagnostic artifacts. The pinned official Meta code revision is `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; HF Base weights are revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, full SHA256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`. Every imported official source file and every used checkpoint tensor is checked. [Pinned official encoder](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py), [official model](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415).

## Four-material ordering and actual adapter feasibility

The balanced four-material experiment is complete. Immediate mixing gave better stucco, soil and cotton train/separate-region results than the tested gradual introduction/replay schedule, with exactly 300 updates per material in both. That gradual schedule exhausted the earlier crops' quotas and ended with brick-only updates, forgetting much of earlier stucco/soil relief. This does not establish that every curriculum fails. Mixed mean training detail correlation was 0.7466, so the predeclared below-0.7 wider-model condition did not trigger; individual soil and brick scores still fell below 0.7. Brick's separate-region amplitude was only 0.1033 of its reference despite 0.9371 on training. See the [complete fitting report](material-fitting-2026-10-07.md).

`scripts/probe_material_adapters.py` then completed two real MPS optimizer updates through the pinned official DINOv2 Base implementation, with rank-8 updates on all 12 attention QKV and attention-output projections: **442,368 adapter parameters plus the 328,957-parameter native material head**. It reused the exact previously learned frozen-feature head, whose nonzero output weights allow adapter gradients immediately. A new compatible head subclass preserves all forward operators and state keys while accepting feature gradients; the prior frozen diagnostic's detached-feature guard remains unchanged. The learned head's file hash and schema, encoder weights, imported official sources and selected crop hashes were checked.

LoRA A uses Kaiming initialization and B starts at zero, with no adapter weight decay. Every B received a finite nonzero learning gradient and changed on step one; A remained unchanged until step two, when every A also received a finite nonzero gradient and changed. The exact original pretrained tensors stayed frozen and fingerprint-identical. The native 1024 RGB/detail branch and raw UInt16 displacement target remain unchanged; only the contextual encoder works at 518. A checkpoint stores adapters plus the head, not another copy of the pretrained Base.

The complete probe took 6.164 seconds including checkpoint loading and fingerprints. The first synchronized update took 2.943 seconds with initial kernels; the second took 0.394 seconds. Sampled Metal driver peak was **4,792,745,984 bytes**, and process peak RSS was 1,135,230,976 bytes; these overlapping measures must not be added. AdamW's allocated tensor state occupied 6,171,064 bytes. The saved adapter/head checkpoint is 3,122,327 bytes. Soft guards check time and driver allocations before forward, after forward and after optimizer, including the final update; they cannot predict transient allocation peaks or interrupt a running Metal operation. All 15 focused CPU tests pass, including differentiation, exact forward equivalence, all guard stages and concurrent-file-change rejection.

This established local **Base adapter learning feasibility**, not improved material maps, whole-encoder fine-tuning capacity or GIANT capacity. The subsequent `scripts/diagnose_material_adaptation.py` quality comparison completed 1,200 updates per variant with identical checked starting heads, balanced four-material updates and exact shared supervision. Initial validation candidates remained eligible when updates worsened the declared objective. No new pretrained model download was required. [PEFT custom/vision model guidance](https://huggingface.co/docs/peft/main/en/developer_guides/custom_models), [original LoRA method](https://arxiv.org/abs/2106.09685).

The locally retained probe report (`dinov2-base-adapter-feasibility-01/summary.json`) and independently reviewed source/test evidence are preserved locally alongside the small checkpoint. No probe weights are selected in Texture Studio.


## Actual adapter quality result

The selected frozen control scored 1.545287 at step 1200, versus 1.571060 for selected rank-8 LoRA at step 800 on disjoint regions of the same four known materials: **LoRA was 1.668% worse**. The final LoRA step 1200 scored 1.624575, 5.131% worse than frozen. Runtime was 203.97 versus 419.33 seconds; sampled driver peaks were 2.201 versus 5.430 GB. The exact original Base, native UInt16 targets and selected source files remained unchanged. Both variants had the same initial predictions and exactly 300 updates per training crop.

Small selected cotton/brick improvements did not overcome stucco/soil regressions. Brick-region height amplitude remained only 10.4% of the target for frozen and 12.2% for selected LoRA. The locally retained quality review (`four-material-adaptation-01/quality-review.json`) distinguishes initial, selected and final results; the locally retained comparison figure (`four-material-adaptation-01/final-fixed-gain-comparison.png`) uses final weights and a fixed display gain. There is no production promotion or fresh-material/raw-camera quality claim.

The immediate next experiment should improve native-head region fitting using both prepared training corners, before larger adapters or encoders. A broader balanced source fit and a separately tested DA3 backbone remain possible comparisons, with the same raw-height supervision and gradient/memory checks. This run does not establish GIANT training capacity or imply that every LoRA configuration will fail.
