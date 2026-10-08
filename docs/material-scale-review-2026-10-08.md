# Material scale and reference review

## Verified current data

The publisher-native 1K collection now contains **97 complete materials**: 93 Poly Haven sets and ambientCG Wicker013, Snow013, Snow014 and Snow015. Every material includes diffuse, displacement, OpenGL normal and roughness. All displacement maps have original UInt16 samples. Normal and roughness retain their real source precision; ambientCG roughness and diffuse are UInt8. No local rescaling, padding, gamma conversion or min/max stretching was applied.

Original downloads live in `/opt/ipde/material-dataset/sources-1k`. The separate whole-map dataset is `/opt/ipde/material-dataset-native-1k-whole`. All 97 samples pass independent decoded comparison against every parent map. Preparation used 12 CPU workers and light PNG compression. Downloading uses eight concurrent material sets by default. Existing 4K sources and the previous 2048 dataset remain intact.

Most maps are 1024 × 1024. Five Poly Haven originals have different actual dimensions: cotton jersey 1024 × 1025, jersey melange 1024 × 973, scuba suede 1024 × 1001, polar fleece 1024 × 1017 and wool boucle 1024 × 1269. They remain complete and unchanged. A publisher's “1K” label does not establish a square model grid. PBRnxt's whole-map input requires each dimension to be a multiple of 64; these five need a different suitable architecture or an explicitly chosen real crop. They must never be silently padded, stretched or described as square 1K inputs.

The whole-map preparer uses `--max-edge 2048` as an allowed source bound, **not** as a training size. Actual sample dimensions and source rectangles are recorded independently. The fitting dataset has no reserved validation materials. This tests fitting on known materials and does not demonstrate generalization.

## Reproducible acquisition and preparation

Use new report/output paths. Existing publisher files are reused only after checksum verification; changed files are left untouched.

```sh
.venv/bin/python scripts/download_material_resolution.py \
  --dataset /opt/ipde/material-dataset-2048 --source-links \
  --resolution 1k --download --download-workers 8 \
  --destination /opt/ipde/material-dataset/sources-1k \
  --max-bytes 2147483648 --report out/material-training/my-native-1k-download.json

.venv/bin/python scripts/download_ambientcg_material_resolution.py \
  --assets Wicker013,Snow013,Snow014,Snow015 --resolution 1K --download \
  --destination /opt/ipde/material-dataset/sources-1k \
  --evidence out/material-training/ambientcg-native-1k \
  --report out/material-training/my-ambientcg-native-1k-report.json \
  --max-bytes 134217728

.venv/bin/python scripts/audit_material_sources.py \
  --sources /opt/ipde/material-dataset/sources-1k \
  --api-cache out/material-training/polyhaven-1k-api-cache \
  --output out/material-training/my-native-1k-audit.json

.venv/bin/python scripts/material_dataset.py prepare \
  --sources /opt/ipde/material-dataset/sources-1k \
  --output /opt/ipde/my-native-1k-whole-dataset \
  --whole-maps --max-edge 2048 --validation-fraction 0 --approve \
  --published-source-audit out/material-training/my-native-1k-audit.json \
  --package-source-audit out/material-training/ambientcg-native-1k/wicker013/wicker013-1k-verified-package.json \
  --package-source-audit out/material-training/ambientcg-native-1k/snow013/snow013-1k-verified-package.json \
  --package-source-audit out/material-training/ambientcg-native-1k/snow014/snow014-1k-verified-package.json \
  --package-source-audit out/material-training/ambientcg-native-1k/snow015/snow015-1k-verified-package.json
```

`--source-links` selects exact Poly Haven asset URLs from the curated dataset, even when old parents were edited and cannot carry a published-file checksum claim. It does not certify those old parents. Newly obtained originals still require the provider's complete MD5/byte count and matching actual paired PNG headers. ambientCG is handled separately through verified original ZIP members and a per-asset photogrammetry declaration. Bare provider dimension fields are retained without guessing meters.

## Whole-map base result

A **no-training** native 1024 × 1024 Metal evaluation of the exact pinned complete PBRnxt base on published whole stucco and brick maps completed successfully. It created no optimizer or new checkpoint and changed no weights. The saved report distinguishes `training_crop_size: null`, actual inspection dimensions and whole-map policy.

The source stucco reference has clear pits and chips; the base prediction remains shallow and blurred, with border artifacts. The brick prediction has recognizable courses but shallow relief and false local bumps. Whole-map context alone therefore did not qualify this experimental 1:1 adaptation for production. This is a visual finding, not acceptance or rejection based only on reference error. The two previous failed fitting checkpoints were deleted at the user's request; their small experiment/provenance records remain.

Open `out/material-training/pbrnxt-published-whole-1k-base/review-manifest.json` in Material Review. The raw Float32 reference EXRs contain the literal UInt16 source codes divided by 65535, with no gamma or range adjustment. `*-relief-8x.png` files are explicitly amplified **display diagnostics** and may clip; they are not training targets or lossless height exports.

A separate **native 2048 × 2048 whole-stucco inference** also completed after fixing evaluation to use an inference budget rather than a backward-pass training estimate. It used the verified publisher 2K original with no resizing or padding, no gradients or optimizer, and a positive 53 GiB Metal allocation cap within a 56 GiB total budget. Evaluation took 49.45 seconds including model loading and exports; 0.1-second memory samples observed 51.34 GiB maximum Metal driver allocation. These samples are not a guaranteed instantaneous peak. The 2K prediction still has shallow, blurred relief and dark border artifacts. This establishes that this one inference ran on this Mac, **not** that native 2K training fits or that the model is production-ready.

The separate paired dataset is `/opt/ipde/material-dataset-published-2k-whole` (two complete materials, both independently verified against parents). The 2K review is `out/material-training/pbrnxt-published-whole-2k-base/review-manifest.json`; memory evidence is `out/material-training/pbrnxt-published-whole-2k-base-memory.json`. Training retains the measured forward/backward resource check. Both evaluation commands now freeze parameters and omit optimizer/scheduler allocation; unmeasured inference peak estimates are explicitly unknown.

Review now pins the real source reference next to predictions, reports actual stored precision, retains checkpoint/source identities, and provides **Open Full Source Displacement Map** without inference. Comparisons restore their own display settings; a new comparison or full-source popout starts at neutral display contrast so another review's settings cannot wash out the reference. Original 4K maps remain directly inspectable through recorded parent paths in crop/checkpoint reviews.

## Carry forward to roughness and normals

The intended next scale experiment combines publisher-native whole maps for context with real, paired higher-resolution crops for detail. Publisher 2K sources with 1K crops are the ordinary detail view; 8K/16K detail crops are explicitly curated exceptions. Keep the diffuse/height/roughness/normal UV rectangles identical, preserve their original precision, and record source resolution, crop coordinates and map convention. Balance materials and views rather than letting materials with more crops dominate. Whether this improves fitting still needs a controlled experiment; the Flux diffuse example motivates it but does not establish height-model success.

For direct PBRnxt/SPAN predictors there is no diffusion denoising sampler. For a future diffusion candidate, record the compatible scheduler, steps, noise schedule and guidance as part of the output identity. See the [Diffusers scheduler documentation](https://huggingface.co/docs/diffusers/using-diffusers/schedulers).

Height-derived OpenGL normals describe the relief in that height map; a supplied normal can contain finer features absent from height. Roughness is a separate material property and cannot be recovered reliably merely by differentiating height. Preserve both real reference maps for future supervised comparisons. Do not carry a configuration into those tasks just because its fitting loss decreased.
