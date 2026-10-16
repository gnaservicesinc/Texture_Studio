# Phone photo to usable material color

Updated 2026-10-16 after the real-capture dataset survey. This root document is
the current plan; the earlier conservative proposal is historical.

The user drops in a decent phone photograph of a supported material. We propose
a useful color/diffuse texture automatically, with optional controls for repair,
lighting, framing and generated detail. This is local generative AI: adding
plausible surface detail is part of the goal. The result should retain the
recognizable arrangement and character of the photographed surface while being
more useful as a material. It is not a scientific reflectance measurement.

## Current implementation

The first implemented milestone is **capture data preparation**, including the
native Factory **Materials → Synthetic Photos…** generator described in
[the Factory feature documentation](https://github.com/gnaservicesinc/LoomFactory/blob/main/Docs/SyntheticPhotoData.md).
It supports all materials or a seeded percentage per category, native 1K crops
from higher resolutions, independent effects, saved recipes and data review.
The tools
in `tools/photo_capture_data/` acquire bounded, identified samples, preserve the
downloaded bytes, make exact native 1K crop derivatives, and prepare scene-based
diagnostic splits. They do not train a model or change Factory's source project.
Actual examples include noisy/clean phone photographs of carpet and fixed-camera
photographs under changing light. Findings, rights and limitations are recorded
in [PhotoCaptureDatasets.md](docs/PhotoCaptureDatasets.md).

Factory now also has **Photo Enhancers…**: separate native Repair and Lighting
RGB residual training, pair import, exact checkpoint continuation, bounded
sessions, Save & Stop, and CPU A/B testing of saved exports. Texture Studio has
manual stage controls after framing; each enabled stage uses a checked, fixed
build-time RGB product before material prediction. The new default has broad
manual lighting balance off; explicit saved settings remain intact. Useful crop/lens/HDR/companion handling is retained;
the old material-making flow is not treated as a learned cleanup solution.

No quality-qualified photo enhancer weights are included. A short native1K diagnostic performed four updates across three SIDD scenes to verify execution, not learned quality. No replacement app is installed and no competing GPU work starts
while the user's material training is active.

## Andrew's next steps for the first usable version

The complete training and inclusion guide is
[Factory Photo Enhancer Training](https://github.com/gnaservicesinc/LoomFactory/blob/main/Docs/PhotoEnhancerTraining.md).

1. Finish the current material job, then install the prepared Factory Release.
2. In **Synthetic Photos…**, select **Repair**, choose all or a random percentage
   per category, and keep native1K crops from higher tiers. Review the generated
   pairs. Import that manifest from **Photo Enhancers…**, Repair stage.
3. Start Compact NAF width32, 1,000,515 parameters, native1K, LR0.0001, batch1,
   WD0, 30 minutes. These are explicit starting settings, not a proven winner.
   The actual model must pass the built-in native1K/Core ML CPU parity check
   before learning. The short native1K update/resume/parity checks passed after
   the GPU lease became free. Long-run stability and appearance still need pilots.
4. Test its best export on unfamiliar real photos and on already-good inputs.
   Use **Test Saved Enhancer…** for a native1K CPU A/B. Reject halos, wiped grain,
   changed cracks/weaves, and identity failures. Extend/continue only if useful.
   The six available SIDD pairs are a diagnostic across three scenes, not a
   sufficient category qualification corpus; import them as Repair only.
5. Repeat with the **Lighting** synthesis preset and separate weights. Add
   artist-reviewed real neutral references for hard shadows/hotspots; alternate
   lit room captures are not automatically albedo targets. Compare both stages
   together with generated height/normal/roughness under moving light.
6. Include your material-model package with `TEXTURE_LOOM_PACKAGE` and reviewed
   photo exports with `PHOTO_REPAIR_MODEL` / `PHOTO_LIGHTING_MODEL` when building Studio. Manual toggles/strengths sit after crop/lens setup.
   First release: opaque SDR1K, finite corrections, no hidden tiling or scaling;
   exact model hashes are pinned in recipes. No weights means no enabled stage.
7. After useful Repair/Lighting, add optional generated microdetail with its own
   strength/seed, then qualify carpet types. Alpha rope/nets require a separate
   opacity/foreground product. Keep the category classifier/router optional
   until real labels show it helps. Do not merge Ground/Small Particles yet.

The first trainer samples imported pairs uniformly per epoch. The proposed
60/30/10 real/synthetic/identity mix is an unimplemented experiment, not a
reported feature. Geometry stays programmatic initially, with one composed warp;
a learned reframing network is unnecessary for the first usable release.

## Product flow

1. **Drop photo.** Preserve the original. Detect a likely supported category and
   separate issues: noise, mild blur, lighting gradients, local shadows, glare,
   clipping and perspective. A multi-label quality estimate is more appropriate
   than choosing one defect. Category and feature choices remain editable.
2. **Propose a result.** Use automatic defaults for enabled capabilities. Start
   with optional framing/lens correction, then restoration, lighting correction
   and category-aware detail generation. Let a good photo bypass unnecessary
   stages. The first release can expose manual toggles before its automatic
   quality router is trained.
3. **Inspect in context.** Original/result wipe, native 100% inspection and
   material appearance under moving light. Each feature has a bypass and
   strength; generated detail can be disabled independently. The normal, height
   and roughness models remain separate products with their own controls.
4. **Export.** Save a derivative and a reproducible receipt: source hash, crop or
   warp, transfer functions, model identities, settings and generation seed.
   Keep the source and previous versions available for A/B comparison.

The UI should call the detail option **Generate detail**, not imply it recovered
information that was actually observed. A soft or clipped region can receive
plausible generated texture when enabled. Ask for a retake when it is so severe
that the surface identity cannot be retained reliably.

## Local network approach

| Stage | First approach | Constraint to establish with measurements |
| --- | --- | --- |
| Quality routing | Manual toggles first; later one small multi-label estimator with region masks | Do not require a separate heavyweight classifier for every defect. |
| Geometry | Camera metadata/calibration where available, optional estimated lens model and four-corner framing | Compose transforms into one explicit resampling of a derivative. No warp for a photo that does not need it. |
| Noise and mild blur | Shared RGB residual restoration network; compare NAF-style and other compact candidates around 1–4M parameters | These are starting size hypotheses, not measured runtimes. Clean inputs need an identity path. |
| Uneven lighting | Small context branch predicting a smooth colored illumination field plus a separate localized correction | Apply context to the native-resolution detail carrier. Do not replace the whole photo with a low-resolution albedo image. |
| Generated detail | Optional category-conditioned residual generator or small category adapter, initially around 1M additional parameters | Preserve large cracks, brick boundaries, weave direction and patterns; vary plausible microdetail with a recorded seed. |
| Transparent materials | Separate opacity/matting product with RGB foreground cleanup | Rope and net holes cannot be inferred by treating dark pixels as transparent. |

Test a deterministic residual baseline first. For generation, compare a
conditional stochastic residual model with a distilled few-step approach only
if it improves real surface appearance within the local time/memory budget.
Using a large teacher during development is optional and subject to its artifact
license; shipping must remain local and small. Do not assume an existing material
Compact model or its parameter count transfers directly to RGB restoration.

Small changes mean controlled correction amplitude and preserved structure,
not a hard percentage of pixels allowed to differ. Noise or a lighting gradient
can cover every pixel. Generation is separately adjustable because it has a
different purpose from suppressing camera noise. Stage ordering should be
tested jointly: fixing lighting after sharpening can expose halos, and detail
generation must not reintroduce the shadows that an earlier stage removed.

Training objectives can include paired RGB reconstruction for repair, gradient
and structural terms, clean-input identity, and illumination smoothness. Detail
generation may add perceptual or adversarial objectives in an explicitly
separate experiment. Select them by useful appearance and artifact inspection;
matching a target or maximizing sharpness alone does not decide usability.

## Data first: repeated photographs

We already have substantial PBR assets. The missing supervision is **the same
surface photographed at varying quality**, not another collection of PBR maps.

* Start noise/restoration with real SIDD noisy/clean pairs, including the observed
  carpet scene. NIND is a promising additional source of repeated ISO captures.
* Use the author-published RealBlur capture pairs for blur/sharp supervision.
  Its HF repository also contains synthetic BSD data; keep those sources distinct.
* Use MIT Multi-Illumination fixed-view captures to study lighting invariance and
  directional shadows. A different lighting direction is not itself a neutral
  diffuse target. Never silently teach one arbitrary lit photograph as albedo.
* Existing authored color textures can support controlled degradation and detail
  priors as secondary data. They do not replace real camera pairs. Teacher
  albedo estimates and artist edits need separate provenance and human review.
* Only request a small targeted phone session after these baselines identify a
  specific gap, such as a carpet pile type, specular hotspot, or alpha boundary.
  Do not ask the user to build a large photo corpus before using available data.

Keep all cameras, ISO settings, burst frames, crops, source tiers and lighting
variants of a physical surface/session in the same split. Multi-Illumination
needs grouping by room/building where surfaces repeat across viewpoints. Never
inflate independent-surface counts by counting crops as new materials. The
current six SIDD samples cover three physical scenes and are diagnostic-only.

## Category scope

Build a category-by-capability support matrix from reviewed examples. Do not
announce support for every existing board category because one generic denoiser
worked. Each claimed feature needs an identity test and several independently
held-out physical surfaces in that category.

**Carpet** is the first proposed new category: begin with short-pile, opaque,
reasonably front-facing photos. Separate tests for woven/patterned carpet and
thick pile, where self-shadow and true fiber depth complicate illumination
removal. One SIDD carpet scene is useful evidence for data availability, not a
qualified carpet model.

**Rope and nets with alpha** are a later explicit capability. Require foreground
RGB plus opacity targets, backgrounds that vary for the same foreground,
foreground color decontamination, premultiplication rules and transparent-pixel
loss masks. Test fine strands over light and dark backgrounds. Do not add them
to an opaque texture training job or reuse normal-map RGB semantics for alpha.

**Ground and Small Particles** retain their labels for now. Test a shared family
with balanced sampling and optional category conditioning before merging them.
More examples increase training work, not automatically network parameter count.
Merge only if appearance and local training cost justify it; preserve child
labels so the choice remains reversible. Different material families can use
different network recipes behind the same product flow.

## Implementation sequence and gates

1. **Real and synthetic data foundation — implemented.** Pinned bounded download, same-frame
   pairing, original checksums, exact native 1K crops, immutable review receipts
   and physical-scene split preparation. Factory also generates controlled
   exposure/blur/noise/illumination and separate reframing pairs from existing
   color maps, with exact native crops, masks, replayable recipes and disk limits.
   A 60% real / 30% synthetic / 10% identity sampler is a starting experiment,
   distinct from the percentage of source materials selected for generation.
   No production training membership is changed.
2. **Restoration baseline — training path implemented; quality pending.** Once GPU capacity is available, measure a
   frozen official RGB restoration model on the sampled photos and clean-input
   controls. A pretrained SIDD model may already have seen these scenes: this
   baseline is a sanity check, not an independent generalization test. Reserve
   genuinely new physical surfaces for subsequent quality claims. Then run one
   bounded native 1K small-network pilot. Record actual
   step/inference time and unified-memory use before selecting model size or
   proposing an overnight run. Existing model architecture names do not prove
   compatibility with these RGB weights.
3. **Lighting baseline.** Review repeated-light captures at matched camera
   coordinates, retaining native detail. Compare an explicit low-frequency
   correction with a learned illumination proposal. Use artist-reviewed neutral
   references for the final target; licensing must permit the intended training.
4. **Controlled detail generation.** Add a toggle, strength and seed. Compare
   restoration alone against restoration plus generation on unfamiliar surfaces.
   Look for false cracks, repetitive fibers, ringing, fabricated text and changed
   patterns. Plausible added grain is allowed; losing the defining shape is not.
5. **Native product integration — baseline implemented.** Repair/Lighting use a
   distinct residual RGB constructor, native1K targets, input/output contracts,
   saved-weight identities, cancellation and exact continuation. Studio has an
   explicit optional build payload and manual controls; material package-v1 is
   unchanged. Photo LoRA/Workshop support is intentionally unsupported in v1.
   CPU export/product/split checks and the short native1K update/parity test
   pass. Long-run stability and useful learned quality still require pilot runs.
6. **Photo-drop workflow.** Integrate automatic defaults, independent controls,
   A/B review and Blender export, then qualify supported categories. Add a
   trained router when real decision labels exist. Alpha remains a separate gate.

Useful appearance under moving environmental light is the main acceptance test.
Inspect native grain and whole materials, not just thumbnails or training loss.
Numerical diagnostics reveal clipping, unexpected color shifts, seams and
identity failures; they do not replace artistic judgment. Record user decisions
alongside each candidate, including cases where different height and normal
details work well together. Keep material-map settings fixed when comparing
color correction so the comparison isolates that change.

## Prior model references worth retaining

[Official NAFNet](https://github.com/megvii-research/NAFNet) is the first RGB
restoration baseline. [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html)
is relevant where independently noisy aligned repeats exist, with its noise
assumptions checked. [Marigold IID lighting](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1)
is an optional lighting teacher/reference, not the proposed small shipping model.
[AnyCalib](https://huggingface.co/javrtg/AnyCalib) is a geometry research candidate.
Check exact weight licenses and local numerical behavior before using any port.

The next learned-quality milestone is a visible restoration A/B on real photos,
followed by a lighting/detail composition review. Data acquisition tests establish
data integrity, not that the resulting texture model has learned useful quality.

## Current material comparison, separate from photo enhancer quality

On 2026-10-16 Andrew's Compact cardboard depth result retains much more fine
texture than the compared Pyramid output, while retaining the corrugated shape.
Use Compact as the stronger starting point for this material and the first RGB
repair experiments. This user appearance judgment does not prove every category
must use one architecture, or qualify a new photo enhancer. Keep alternative
models and judged Blender examples available for A/B comparison.
