# Texture-height model vetting

Checked against primary sources and native Metal probes on 2026-10-08. The previous DINOv2 material-head experiment is retired. Dataset preparation, saved review and export remain available. Studio does not automatically substitute DA3 or activate an unreviewed replacement.

## Current experimental base: complete PBRnxt

[PBRnxt](https://github.com/aaf6aa/PBRnxt) is trained for flat diffuse-to-material decomposition, with albedo, OpenGL normal, roughness and displacement outputs. Its source is MIT with retained Apache-2.0/MIT component notices. The pinned checkpoint is `pbrnxt_402236.pth`, revision `73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35`, 349,493,406 bytes, SHA256 `3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d`.

The complete network has **86,763,756 pretrained parameters**. All 2,896 state keys load strictly. Keeping only its 66.52M-parameter intermediate generator produces a repeating grid, not usable displacement; CPU and Metal agree. The final pretrained RRDB branches are essential. The native-scale adapter retains every trained convolution and output head, omits the two nearest-neighbor enlargements per final branch, and removes the wrapper's artificial image borders. It disables stochastic evaluation noise for repeatable comparisons. This is an explicit **experimental 1:1 adaptation**, not the unchanged published 4x model.

At native 512, the complete mapping responds to photographed stucco chips and brick pores. Broad relief can be shallow or inverted and boundaries have artifacts. A small supervised fitting trial is appropriate; production quality has not been established. Refinement first trains the existing final height branch (5,059,745 parameters) while preserving the complete pretrained material core. A saved delta is not a standalone small model: it requires the exact 349MB pretrained base and its notices.

Upstream [preprocessing](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/scripts/dataset_preprocess.py) writes UInt8 JPEG training references. Our refinement bypasses it, retaining the original **UInt16 linear numeric height codes**, mapped to Float32 by division by65535 only. No target gamma, min/max stretching, source padding or rescaling occurs. Float32 computation/export preserves those codes' distinctions; it cannot add missing source information.

## Actual training size and resources

The active `/opt/ipde/material-dataset-2048` dataset contains genuine 2048 paired maps. Each update in the first trial sees a **real 1024 × 1024 crop** from those maps. Dataset size and model input size are different recorded fields. Joint flips/90-degree rotations and modest exposure changes affect derived training tensors only. The height loss excludes 64 boundary pixels; the input still contains 1024 real pixels per axis. Original PNGs remain unchanged.

A complete native 1024 forward/backward on the 64GiB M2 Max took 10.40/7.64 seconds and 18.67GiB sampled Metal driver memory. The fitting run settles around 25GiB including optimizer/workspace caching. Memory estimates, CPU cache, driver cap and actual native crop are recorded. The driver limit follows the selected unified-memory budget with room for host cache and macOS. Native 2048 training is not qualified: its activation estimate exceeds this Mac's comfortable budget. Choose a lower real crop; never manufacture border pixels or upscale a 1K target.

The reusable research entry point is [train_material_pbrnxt.py](../scripts/train_material_pbrnxt.py). It supports train-from-base, refine-from-checkpoint and evaluation, saves exact identities and raw Float32 EXRs, and produces labeled native review manifests. It does not automatically mark results production-ready or change Studio's model. The native training interface stays paused while this replacement is evaluated.

```sh
.venv/bin/python scripts/train_material_pbrnxt.py train \
  --dataset /opt/ipde/material-dataset-2048 \
  --output out/material-training/my-pbrnxt-run \
  --download --size 1024 --updates 194 --memory-gib 56

.venv/bin/python scripts/train_material_pbrnxt.py refine \
  --dataset /opt/ipde/material-dataset-2048 \
  --output out/material-training/my-pbrnxt-refinement \
  --checkpoint out/material-training/my-pbrnxt-run/checkpoint.latest.pt \
  --size 1024 --updates 194 --learning-rate 0.00001 --memory-gib 56
```

Use spaces between each option and its value (for example `--size 1024`). Use new output folders; previous runs are preserved. Base source/licenses/weights can be downloaded with `--download`, or supplied through both `--source-dir` and `--weights`. No models, datasets or generated maps enter Git.

## Excluded or secondary candidates

- **DA3 / DA3MONO:** camera-depth estimators are not the chosen texture-height training base. The requested rejection is honored; no DA3 material training ran. An explicit Studio camera-depth option is a separate workflow.
- **DeepBump:** texture-trained normal-to-height baseline, GPL-3.0, about 26.7MB. It remains a comparison reference; its small normal network and integration are not substituted for a qualified complete material model. [Official source](https://github.com/HugoTini/DeepBump).
- **MaterialPalette decomposition:** a substantial pretrained BRDF network, but no learned displacement output. The [actual architecture](https://github.com/astra-vision/MaterialPalette/blob/0ddd360467cce8879c9772fb791924b0dc0b9d9b/capture/source/model.py) declares Monodepth2 noncommercial restrictions despite the repository's MIT label. Exclude it pending license clarification. Its stock loaders/exporters also reduce maps to 8-bit and resize inference to 512.
- **CHORD:** relevant material decomposition, but its research-only model license is unsuitable as this GPL production default. [Official source](https://github.com/ubisoft/ubisoft-laforge-chord).
- **Generated material diffusion:** image-prompt generation can change photographed texture identity. It is not this initial supervised enhancement path.

## What counts as improvement

Inspect original-resolution maps at 100% and displaced Blender surfaces under neutral/grazing light. Check bumps in the right places, inversion, false relief from shadows, grain, halos, seams and noise. Always compare the exact pretrained base and checkpoint on the same pixels with durable labels. Raw EXRs retain unclipped values; explicitly labeled shared display contrast affects previews only.

The first short trial uses known stucco/brick sources to test fitting. The small automatic same-source check set measures fitting progress, not unseen-material generalization. Fresh photographs and additional independently captured materials must test generalization before production selection. A score improvement alone is not acceptance.

[Prodigy](https://github.com/konstmish/prodigy) adapts learning-rate estimates; it is not an overfitting detector. Scheduled AdamW, conservative refinement, paired-data quality, realistic illumination changes, material balancing and visual checkpoint selection address the actual risks. Compare another optimizer only after the task and baseline work.
