# Real photo-pair preparation

These are development data tools for the local photo-to-diffuse effort, not a
bundled Python runtime or a trainer. The input is repeated real camera captures
with their matching cleaner references. PBR maps are not a substitute for this
data. See [the product plan](../../PhotoToDiffusePlan.md).

`sample_sidd.py` downloads a bounded sample from the pinned SIDD-CC sRGB release.
It pairs each noisy frame with its own reference, retains upstream originals,
checks file size and SHA256, and writes aligned **native-pixel** 1024 crops plus
a side-by-side review. It does not resize, re-encode or change the downloaded
originals. Crop derivatives preserve their decoded pixel bytes and ICC profile.

Requires Python 3.11+ and Pillow. No PyTorch, dataset framework, GPU job or model
weights are used. Do not install Python into a shipping app.

```sh
python3 tools/photo_capture_data/sample_sidd.py \
  --output "$HOME/Library/Application Support/Loom Factory/PhotoCaptureData/carpet" \
  --scene-ids 008 --variants 2 --max-mib 128

python3 -m unittest discover -s tools/photo_capture_data -p 'test_*.py' -v
```

Scene `008` is a photographed patterned carpet in the surveyed release. No claim
is made that its center crop covers all carpet types or qualifies a model.
The default selects the first two scenes and two capture conditions per scene.
All quality variants, cameras and burst frames from a **physical scene** share
one group. Captures with different camera/lighting settings are never treated as
perfectly aligned interchangeable targets. Within each supplied pair the noisy
frame and reference retain the dataset's alignment, which still needs visual
inspection before training. Only ten physical scenes underlie these 230 training
pairs; 230 independent surfaces would be an incorrect claim.

Receipts begin with `split: unassigned`, `reviewed: false`, and no material
category. They live outside the source project and do not add Factory image
membership, tags, approval or model assignments. A denoised SIDD reference still
contains scene lighting: it is **not** a shadow-free diffuse/albedo target.
Original SIDD declares MIT terms; SIDD-CC inherits those terms and asks for both
citations. Keep the source links and original attribution with any retained data.

The byte budget covers downloaded originals; derived crops/review images use
additional disk space. Existing originals must pass their upstream checks and
are reused; mismatching files and completed review receipts are never replaced.
On failure preserve the evidence and choose a new output folder for a new review.

## Verify and stage scene-separated diagnostics

```sh
python3 tools/photo_capture_data/prepare_split.py \
  --reviews "$HOME/Library/Application Support/Loom Factory/PhotoDiffuseResearch-2026-10-16/real-phone-pairs" \
            "$HOME/Library/Application Support/Loom Factory/PhotoDiffuseResearch-2026-10-16/carpet-phone-pairs" \
  --validation-groups sidd-scene-002 --test-groups sidd-scene-008 \
  --output "$HOME/Library/Application Support/Loom Factory/PhotoDiffuseResearch-2026-10-16/diagnostic-split.json"
```

This rechecks every original hash and every native crop's actual pixel bytes,
ICC profile and coordinates. A scene's variants cannot cross splits even when
acquired through different reviews. Duplicate captures, missing/overlapping
holdouts, and splits leaving no training scene fail. Output is diagnostic-only,
not approval or a trainer input contract. Existing output is never overwritten.

## Repeated real lighting

`sample_multilum.py` acquires two fixed author-example scenes from MIT
Multi-Illumination under a 64 MiB archive budget, with pinned sizes and SHA256s.
It extracts five explicit lighting directions as unchanged JPEGs, retaining the
archives. The provided mip1 images are 3000×2000 upstream derivatives; they are
not full-resolution RAW. There is no neutral albedo target in this review.
Downloaded originals and extracted files both count toward disk use.

```sh
python3 tools/photo_capture_data/sample_multilum.py \
  --output "$HOME/Library/Application Support/Loom Factory/PhotoCaptureData/repeated-light" \
  --max-mib 64

python3 tools/photo_capture_data/sample_multilum.py \
  --verify-existing "$HOME/Library/Application Support/Loom Factory/PhotoDiffuseResearch-2026-10-16/real-lighting-captures"
```

Original captures are CC-BY per the author. Derived MIDIntrinsics albedos have
different, noncommercial terms and are not acquired by this utility. See the
[dataset findings](../../docs/PhotoCaptureDatasets.md) for sources and limitations.
