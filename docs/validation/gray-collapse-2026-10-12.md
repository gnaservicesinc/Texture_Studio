# Gray checkpoint output: numeric diagnosis and convergence checks

Texture Studio 0.9.22, build 117. Evidence is retained outside the repository at
`~/Library/Application Support/Texture Studio/Validation/gray-collapse-2026-10-12`.

## Saved predictions have lost relief

The user's `Texture Studio Maps 372B11F3` exports contain real low-variation
Float32 predictions. This is visible before display contrast, gamma conversion,
8-bit preview conversion, or EXR import. The source displacement PNG is
byte-identical to `/opt/ipde/sources_mats/Bark Brown 02/bark_brown_02_disp_1k.png`,
SHA-256 `e02b952d984561ca89e5b33e788914d9bda286a4a1914323d0557e910892be98`.
Its original 16-bit scalar codes are compared directly on the same 1024 grid.

Source mean is 0.26107, standard deviation 0.10308, and the lowest MAE achievable
by any spatially constant map is 0.07750 (the source median is 0.24642).

| RGB checkpoint | Prediction mean | Standard deviation | Variation / source | Source MAE | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 369 | 0.15794 | 0.03272 | 31.74% | 0.11645 | -0.00943 |
| Step 738 | 0.64170 | 0.01150 | 11.15% | 0.38160 | 0.05947 |
| Step 1,107 | 0.59414 | 0.00477 | 4.63% | 0.33454 | 0.05059 |
| Step 3,690 | 0.62269 | 0.01091 | 10.59% | 0.36276 | 0.04000 |
| Step 7,011 | 0.55235 | 0.00425 | 4.13% | 0.29358 | 0.04234 |
| Final export, digest 63f7400691dd | 0.54007 | 0.00472 | 4.58% | 0.28165 | 0.04358 |

The alternate export `9af54b5fc760` has 5.46% of source variation and MAE
0.32211. The already-exported DepthPro-refiner step 369 has 1.24% of source
variation and MAE 0.34961. Reading these existing files does not run DepthPro.
Independent Python numeric decoding and the new native agreement reader agree
within Float32 source-code rounding.

## Data and graph audit

The RGB run used 369 maps, 20 epochs/7,380 updates, width 32, native 1024,
constant learning rate 0.001, one map per update, and gradient cap 1. Validation
was disabled. The DepthPro run used the same 369-map count at 512 and stopped
after 1,116 updates; its validation was also disabled. Their original Chimpmonk
prepared crops are no longer available, so the precise training distribution
cannot be reconstructed from the present Wolf dataset.

Current scalar decoding uses original integer codes divided by their code
maximum, ignores scalar PNG gamma/alpha for supervision, and retains paired
spatial coordinates. The bark diffuse is sRGB, matching its legacy checkpoint's
sRGB tensor contract. Full-backbone gradients and optimizer updates have numeric
regression checks. No missing-gradient, target-gamma, or bark comparison-transfer
defect was established by this audit.

The broader run's exact cause remains unproven. Mixed material calibration,
limited context, exposure per subject, loss behavior, and the training schedule
are possible contributors. The experiments below show that this native model
and the saved gray weights can recover a known target; they do not select one
of those possible causes or qualify a general material model.

## Repeated-update evidence

New always-on tests require RGB scalar and one-channel compact graphs to learn
a deterministic nonconstant scalar relationship over 150 updates. They check
MAE against the optimal constant, target correlation, and recovered variation.
The one-channel test supplies an analytic field; it never invokes DepthPro.

An opt-in real-material test repeatedly updates the exact top-left 256 crop of
the original bark diffuse/height pair, without resizing, padding, or changing
source files. Its first fresh-width-16 trial at 0.001 for 200 updates failed the
quality threshold: MAE 0.05277 versus constant baseline 0.05287. Structure had
partly recovered (correlation 0.595), but its absolute level still oscillated.
That failure is retained in `focused-tests.log`, and the thresholds were not
weakened to pass it.

Continuing that fresh probe for 800 more updates at 0.0001 reached MAE 0.00461,
correlation 0.99621, and 96.19% of source variation. A separate copy of the
actual step-7,011 width-32 checkpoint, trained on the same known crop for 800
updates at 0.0001, reached MAE 0.00338, correlation 0.99873, and 99.60% of source
variation. Its initial crop MAE was 0.29928 and variation 6.49% of source.

All these crops are training pairs, not held-out materials. The probe weights
remain outside the model library; no original checkpoint is replaced or
activated. Receipts and raw Float32 snapshots retain source/checkpoint hashes,
crop coordinates, input transfer, settings, and measured values. New probe
checkpoints explicitly identify their known-crop scope and lack of quality
qualification.

## Full-size recovery after the crop check

The crop-recovered model did not initially recover the complete 1024 field:
full-image correlation was only 0.05132 and MAE was 0.08678. Thus the successful
256 check was not enough to present as a full-size improvement. A separate
copy of the 400-update crop probe then received 200 updates on the **complete
original 1024 diffuse/height pair** at 0.0001. Its full-image MAE reached
0.04533, correlation 0.81803, and variation 60.92% of source. The output retains
visible bark relief, though finer detail and large relief amplitudes remain
less accurate than the source.

A fresh production RGB inference on the original 1024 diffuse reproduced this
result and passed explicit MAE/baseline, correlation, and variation criteria.
The production/checkpoint/source identity, native dimensions, no-resize path,
and absence of DepthPro setup were verified. This checkpoint is still trained
on this same bark map. It is a usable known-material recovery probe, not a
validated general replacement for the 369-map model.

The verified native model package is in
`full-native-production/full-source/model-package` under the evidence directory.
`fresh-full-source-review.json` compares the original source, the old step-7,011
gray export, and the fresh production output. `full-source-before-after.png`
uses the same raw 0–1 display range for all three panels; contrast has not been
individually expanded to make the recovery look stronger.

## Application changes

Comparison now shows raw source MAE, the best constant baseline, Pearson
correlation, and output/source variation for height and roughness. The
below-10%-variation indicator is an inspection heuristic. It does not gate a
run, normalize a target, alter output values, or certify unseen-material
accuracy. Source/prediction-only reviews can show agreement without a diffuse
photo. Canceling or changing the selected output cancels its numeric worker.

A separate comparison bug was fixed: the selected diffuse variant's encoding
is now forwarded to both baseline and checkpoint inference and recorded in
the review manifest. This corrects linear diffuse variants. The supplied bark
uses sRGB, so this change does not repair its existing gray predictions.

RGB checkpoint comparison requires no DepthPro model, download, cache, or
generated input pane. A native regression verifies this with real RGB
inference and a fail-fast DepthPro dependency. Refiner checkpoints keep their
separate one-channel preparation contract. The source height is a comparison
reference, never substituted as a model input.

## Validation

The final focused Release suite passed **40 tests, no skips, zero failures**.
It includes the native production audit of all eight exported EXRs, both
synthetic convergence tests, and a 400-update recovery from the real step-7,011
checkpoint with original source/checkpoint byte identity checks. That repeat
reached MAE 0.00713, correlation 0.99297, and 96.28% of source variation.

The subsequent full-native known-map continuation passed its convergence
check. The final production-inference, comparison, and view-layout batch passed
**14 tests, no skips, zero failures**, including native model-package export and
verification. These additional results are retained in
`full-native-training.log` and `full-native-production.log`.

`make package`, interpreter-free bundle checks, recursive strict code-signature
verification, native build-tool tests, and source/release version agreement
passed. The packaged executable's `--smoke-test` prints **Texture Studio native
smoke passed**. The compiled runtime and source inspection are verified; native
UI automation was unavailable because its pipe failed to start.

The updated built Material Review app was launched with the fresh three-map
manifest. The native installer's `--if-closed` check skipped installation while
the installed Texture Studio remained open. No running installed app was
replaced, and no installed-build smoke or live UI click-through is claimed.

The main executable SHA-256 is
`eae2083b14b1232b6f8b90cacf10f7d2d5f99386ae14503a2bce775284a09963`.
The distribution zip SHA-256 is
`2675fb83ba0850c4dbacb00d85d39456578350bf5e27b494370ecbde201baeed`.
Commands and results are retained in `final-focused-tests.log`, `package.log`,
and `release-receipt.json` in the evidence directory.
The original 200-update quality failure remains available alongside successful
known-crop recovery receipts. A successful focused regression does not qualify
the full 369-map run or novel-material performance.
