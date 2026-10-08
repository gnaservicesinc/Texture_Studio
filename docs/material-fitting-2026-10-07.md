# Material fitting experiments — October 7, 2026

This source tree contains the implementation and documentation. Experiment checkpoints, generated maps, source snapshots, recovery recipes and reports are retained locally and are not included. Record names below identify that local archive.

The recorded losses and correlations below diagnose fitting; they do not rank realistic material quality. The later [material quality workflow](material-quality-workflow.md) uses the photo on actual displaced geometry, shared lighting/strength, clay views, and manual usefulness judgments. The original experimental values are retained as history.

More source materials are not the first intervention. The previous all-material run underfit both training and disjoint validation crops, so this round first checks capacity on repeated examples, then tests a different loss over all 56 materials and a pretrained feature branch. These are experiments; no checkpoint is automatically selected in Texture Studio.

## Repeated native crops

`scripts/diagnose_material_fit.py` trained the existing 218,209-parameter native height network from scratch on one 1024×1024 crop, with no augmentation, 300 updates, batch one, learning rate 0.001 and seed 2307. Both objectives start with identical weights. Inputs retain their actual source precision; UInt16 height codes become Float32 divided by 65535 only in memory. The optional validation image is a non-overlapping region of that same material.

The current L1 objective is centered-height L1 plus 16 times multiscale-gradient L1 and 4 times radius-4 high-pass L1. The alternative combines squared centered-height, multiscale-gradient and high-pass errors in reference-energy units. Detached energy denominators do not rescale targets or outputs. This controlled comparison changes only the objective. Comparison to the earlier all-material pilot also changes update frequency, learning rate and augmentation, so it does not isolate the effect of any one of those choices.

| Repeated training crop | Objective | Height correlation | High-pass detail correlation | Native gradient RMS amplitude / reference |
| --- | --- | ---: | ---: | ---: |
| white_stucco_02 | L1 | 0.9922 | 0.8555 | 0.8423 |
| white_stucco_02 | relative squared | 0.9877 | 0.8927 | 0.8905 |
| farm_soil | L1 | 0.9931 | 0.7602 | 0.6911 |
| farm_soil | relative squared | 0.9927 | 0.8550 | 0.8050 |

The architecture can reproduce these materials' relief. Squared supervision improves the fine-detail fit, while both objectives preserve most of the target's overall amplitude. Nonzero finite upstream gradients and changed encoder weights were verified after the second update, rather than checking only the initially zero output head.

| Disjoint region of known material | Objective | Height correlation | High-pass detail correlation |
| --- | --- | ---: | ---: |
| white_stucco_02 | L1 | 0.8420 | 0.6402 |
| white_stucco_02 | relative squared | 0.8409 | 0.6484 |
| farm_soil | L1 | 0.7646 | 0.5199 |
| farm_soil | relative squared | 0.7910 | 0.5252 |

These separate-region scores are lower than repeated-crop scores. They measure transfer within one familiar material, not a new material or casual photograph. The much broader mixed-material task remains unresolved.

Reports and raw Float32 predictions are in `out/material-training/repeated-fit-stucco-01/` and `repeated-fit-soil-01/`. Their `loss-comparison.png` previews use fixed shared gains for reference and prediction; no numeric output is stretched. Each diagnostic saved about 84 MB including numeric normal/height arrays and exports, with about 2.2 GB sampled Metal driver allocation.

## Longer mixed-material run

`native-height-region-squared-01` warm-started the earlier compatible checkpoint, reset AdamW, and ran 3,000 native-1K updates across all 112 training crops at learning rate 0.001. Validation covered all 56 disjoint known-material regions. It completed in 821.17 seconds including scheduled and final evaluations, with about 2.2 GB sampled Metal driver allocation. Only a bounded set of numeric examples was exported.

| Metric | Starting model, validation | Post-update model, validation | Post-update model, training |
| --- | ---: | ---: | ---: |
| Native gradient L1 improvement over flat height | 1.25% | 4.95% | 5.34% |
| Mean per-sample gradient-energy ratio | 0.143 | 0.237 | 0.250 |
| Prediction standard deviation | 0.00336 | 0.00892 | 0.00918 |
| Target standard deviation | 0.07746 | 0.07746 | 0.06977 |

The mean L1 detail metrics improved, but the reference-energy squared validation objective worsened from 2.9892 to 3.3596, approximately 12.4%. Per-sample energy weighting and ordinary mean L1 answer different questions. Independent selection review therefore retains the starting weights by the declared objective. The new model still flattens most soil/stucco relief and strongly underfits training crops; it is not a production result.

This process started before the trainer's initial-candidate retention guard was added. Its `checkpoint.best.pt` denotes the best *post-update* candidate. `selection-review.json` explicitly compares that candidate with the separately recorded initial validation and immutable initial checkpoint. New runs retain the validated initial candidate automatically and snapshot implementation code. Historical checkpoint bytes and report meanings are preserved; no app model was changed.

Full evidence is under `out/material-training/native-height-region-squared-01/`, including all-sample metrics, frozen dataset metadata, `selection-review.json` and a six-material `known-region-comparison.png` with fixed display gains.

## Pretrained alternatives

The pinned 26.7 MB DeepBump normal predictor completed six native validation crops on CPU. It improves raw reference-normal angular error over a flat normal for three of six crops, but its integrated relative height has weak correspondence to the original displacement. It is a measured material-specific baseline, not a production height recommendation. See [candidate details and actual scores](material-transfer-options-2026-10-07.md).

A medium pretrained encoder plus a native RGB/detail decoder was tested next. Freezing the encoder isolates its feature contribution before adapters or whole-network optimization. LoRA reduces trainable parameter/optimizer storage; activations still require memory. A larger network does not automatically improve material relief, and scene-depth output is not the supervised target for this job.

The actual DINOv2 Base comparison has now completed. It loaded the pinned 86.58M-parameter encoder, kept it frozen, and cached finite normalized patch features at a 518-pixel contextual working size. The supervised RGB branch and height output stay native 1024. The two heads share the exact same initialization and relative-squared objective, each taking 300 updates on stucco. One receives the frozen features and one receives zeros through the same conditioning branch. Both begin with identical 0.5 predictions. The second update verifies nonzero projection gradients for the feature branch, nonzero native-encoder gradients and changed weights; the pretrained encoder remains frozen.

| Stucco comparison | Zero features | Frozen DINOv2 features |
| --- | ---: | ---: |
| Training height correlation | 0.9894 | 0.9880 |
| Training high-pass detail correlation | 0.8925 | 0.9036 |
| Disjoint-region height correlation | 0.8714 | 0.8423 |
| Disjoint-region high-pass detail correlation | 0.6538 | 0.6455 |
| 300-update head training time | 49.77 s | 49.92 s |

The pretrained branch slightly improves repeated-crop fine-detail fitting but does not improve this disjoint region. This is one material, not a mixed-material trial or LoRA. Do not infer that a larger pretrained encoder is either always better or unnecessary. Both heads used about 2.2 GB sampled Metal driver allocation during cached-feature training; this is not a full-encoder fine-tuning memory measurement. Encoder weights occupy 346.35 MB, stored once. The paired run saves about 167 MB of bounded diagnostic outputs and separate native Float32 targets/predictions, with fixed-gain `feature-comparison.png` previews.

The four-material comparison below tests mixed updates against gradual introduction. The later matched adapter comparison is also complete and recorded below. A wider native decoder remains a candidate. More independent materials become the priority after the model adequately fits its existing training relief. Fresh sources then provide a separate generalization check.

## Four-material fitting and ordering

`four-material-fit-01` trained stucco, soil, cotton and brick from scratch at native 1024, with the relative-squared objective, no augmentation, seed 2307 and learning rate 0.001. Both 218,209-parameter models had identical initial weights and exactly 300 updates per training crop, 1,200 total. Mixed updates shuffled all four throughout; gradual introduction added one material at a time and replayed the earlier materials until their equal final update quotas were exhausted. This particular replay schedule consequently ends with a brick-only tail. Its result tests this ordering, rather than establishing that every curriculum is worse.

| Material | Mixed training detail correlation | Gradual training detail correlation | Mixed disjoint-region detail correlation | Gradual disjoint-region detail correlation |
| --- | ---: | ---: | ---: | ---: |
| white_stucco_02 | 0.7297 | 0.2792 | 0.6702 | 0.2473 |
| farm_soil | 0.6331 | 0.4067 | 0.6083 | 0.3036 |
| cotton_jersey | 0.9381 | 0.8379 | 0.9428 | 0.8302 |
| broken_brick_wall | 0.6855 | 0.8810 | 0.1410 | 0.1470 |

Mixed training preserves a mean 0.9064 ratio of predicted to reference height standard deviation. It fits three of the four separate known-material regions substantially better than gradual introduction. Earlier stucco detail in the gradual schedule falls from 0.7319 after update 450 to 0.2792 at completion, consistent with forgetting while later materials dominate. Brick remains a difficult transfer case: the mixed model retains 0.9371 of reference amplitude on its training crop but only 0.1033 on the separate region. These are familiar-material region checks, not fresh-material generalization.

The predeclared conditional wider-model branch triggers only below mean training detail correlation 0.7; the mixed run achieved 0.7466, so that branch did not run. Soil and brick are still below 0.7 individually. This mean threshold does not establish adequate fitting of every material or rule out a useful wider decoder.

Both runs completed in about 197 seconds each, with about 2.2 GB sampled Metal driver allocation. Actual upstream gradients and changed native encoder tensors were checked after the second update. Model plus AdamW checkpoints were atomically saved every 100 updates and each phase; final files retain both states. Automatic CLI resume is not implemented. The complete reports contain per-material phase metrics, while bounded Float32 predictions cover only stucco's train and separate-region crops. All 24 selected manifest/map files remained SHA256-identical even though unrelated dataset additions changed the active index during training.

The locally retained phase plot (`four-material-fit-01/phase-detail-comparison.png`) shows all correlation values, including negative early results. The locally retained stucco comparison (`four-material-fit-01/stucco-comparison.png`) uses fixed shared display gains. Raw numerical arrays are unchanged and retained outside Git. Neither experimental checkpoint is selected in the app.

## Additional sources and published-resolution footprints

The earlier frozen source batch expanded the dataset from 56 to 69 materials: 207 native 1K samples, split into 138 training corners and 69 disjoint known-material validation regions. All 39 new samples have verified published Poly Haven parent-file identities. Every one of the original 168 sample manifests is byte-identical, all 726 original cropped PNGs retain their full-file SHA256 hashes, and the old index entries are unchanged. All 207 samples pass independent decoded source-region comparison. Geometry checks cover 588 train/validation comparisons across the four map roles.

The already supplied published 2K bamboo maps were separated into `sources-2k/` with exact original-file checksums, without resizing or converting numeric values. They join the earlier stucco and soil pilot. `scripts/plan_material_scales.py` checks exact rational surface footprints and records that actual cross-resolution registration remains unverified. All five candidate native 1K corners/center regions for each of these four 2K parents overlap that material's current 4K center validation footprint. They were therefore not assigned to training. A scale experiment needs a separate versioned split across source resolutions; different filenames alone do not make validation independent.

Later-arriving folders were recorded separately from the frozen 69-material batch. The subsequent continuation is documented below. Complete matching diffuse, displacement, normal and roughness source sets can be added to the next ingestion without manual cropping. The new `recreation-69-materials.json.gz` recipe preserves 294 required parent records, exact source identities and crop metadata with no unresolved download origins. The earlier 56-material recipe remains unchanged for historical experiments.

## Genuine adapter feasibility

The pinned DINOv2 Base then passed a two-update MPS LoRA probe: 442,368 rank-8 attention adapter parameters and 328,957 native material-head parameters. Every adapter B received a finite nonzero gradient and changed on the first update; every A did so on the second update. The original pretrained Base tensors and exact raw/native height targets remained unchanged. Actual driver allocation peaked at 4.793 GB, and the second synchronized update took 0.394 seconds. The complete probe, including load and fingerprints, took 6.164 seconds.

This proved an actual adapter learning path fits locally; it did not measure output quality or GIANT/full-network training capacity. The subsequent matched frozen-versus-adapter four-material quality comparison is complete, using the same learned starting head in both variants and retaining the initial known-region candidate if training worsens its objective. The [adapter report](material-transfer-options-2026-10-07.md) records the exact pins, gradient proofs, memory guards and read-only checks.

The user subsequently added three sand examples and additional mud, dirt, rock, brick and ground sources. These stayed outside the 69-material snapshot when initially deferred. The later continuation authorizes preparation of a new frozen source batch; it does not change the four-material trial or historical records.

## Storage and reproducibility

Source parents remain untouched. The compressed `material-dataset/recreation.json.gz` recipe stores source hashes and download evidence, exact crop/split/encoding records, original notes and preparation code. Experiment outputs are bounded; no duplicate full 2K dataset or permanent Float32 copy of all targets was created. Original UInt16 PNGs remain sufficient storage for training with Float32 computation.

The published-resolution pilot retains eight provider-original 2K maps for stucco and soil, about 125 MB; these are downloaded originals, not resized training copies. The full Poly Haven 2K plan is about 3.14 GB. Original 4K parents are still present. A 16.55 MB compressed recipe now resolves all 242 selected 4K parent files, including Wicker's verified AmbientCG archive members. Small experiment records/checkpoints/comparison sheets were separately preserved under `material-dataset/experiments/`, with 30 copies verified by SHA256 (17.69 MB), avoiding duplicate Float32 prediction folders.

Validation for the earlier 56-material experiments passed 134 relevant Python regression tests, Python compilation, shell syntax and Git whitespace checks. The actual MPS optimizer, frozen-encoder gradient path, CPU pretrained baseline, exact Float32 EXR readbacks, provider downloads and idempotent file reuse were exercised separately; that stage left its active 168-sample index unchanged. The later 69-material ingestion and genuine adapter probe passed the additional preservation/gradient checks described above. No experimental model was promoted into the app.

The earlier relevant material suite passed **178 CPU tests**, including the prepared next-cycle adaptation contracts. Independent review found no remaining blocker. Earlier in this work, `make test-native` passed all 63 Xcode tests and `make test-python` passed 410 verification tests; subsequent changes affect material scripts, evidence and documentation. New-script compilation, CLI help and authored-file whitespace checks pass. That locally retained historical validation record (`validation/summary.json`) distinguishes the work completed at that stage from the then-unrun quality trial and deferred sources. See [next-cycle instructions](material-next-cycle.md).


## Completed matched frozen-versus-LoRA quality trial

Both variants completed exactly **1,200 real MPS updates**, balanced at 300 updates per training crop across stucco, soil, cotton and brick. They used the identical checked stucco-trained starting head and identical initial predictions. The initial mean known-region objective was 2.862892 for both. Supervised RGB/height remained native 1024; only the contextual encoder used 518. The selected eight source manifests and 16 input/height files, all target values and the pretrained Base fingerprint remained unchanged. This is familiar-material region evaluation, with a recorded stucco warm-start bias; fresh materials and raw camera photographs were not tested.

| Variant | Final training objective | Selected region objective | Selected step | Final region objective | Runtime | Sampled Metal driver peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen features | 0.698002 | 1.545287 | 1200 | 1.545287 | 203.97 s | 2.201 GB |
| Rank-8 LoRA | 0.714301 | 1.571060 | 800 | 1.624575 | 419.33 s | 5.430 GB |

The selected LoRA checkpoint is **1.668% worse** on the declared equally weighted region objective; the final LoRA checkpoint is 5.131% worse. Runtime is 2.06 times the frozen control. At the selected checkpoints, cotton and brick improve slightly while stucco and soil worsen. Both variants learn their repeated training crops, but neither gives usable brick-region transfer: height amplitude recovery is only 10.4% for frozen and 12.2% for selected LoRA. The fixed-gain stucco comparisons still show softened fine structure. No experimental checkpoint was promoted into Texture Studio.

The locally retained quality review (`four-material-adaptation-01/quality-review.json`) retains per-material height/detail correlations and amplitude ratios for initial, selected and final checkpoints. The locally retained fixed-gain preview (`four-material-adaptation-01/final-fixed-gain-comparison.png`) explicitly shows **final** weights, while labeling the earlier selected LoRA step. The locally retained phase plot (`four-material-adaptation-01/native-detail-phase-comparison.png`) shows every evaluated checkpoint and all four material families. Complete metrics, exact source/code snapshots, small final/selected checkpoints and the compressed console log are preserved in the same local experiment archive. Full numeric Float32 EXR/NPY height and OpenGL-normal predictions remain in the local experiment folder.

This trial does not support increasing adapter rank or model size as the immediate remedy. The next controlled fitting comparison should expose the frozen head to both already-prepared training corners of each material, measure whether brick relief transfers to its disjoint center, and retain native target amplitude/detail. A separate raw-target inspection found brick height standard deviations of 0.022545 in the first training corner, 0.128480 in the second and 0.209493 in the held-out center. The second corner contains deeper joints/cracks, so it narrows the representativeness gap without changing the validation region. These are stored-code statistics and visual observations, not physical measurements. Additional source families can then enter a balanced fit. The 2K provider-original maps still require the surface-footprint split and registration checks described above before joining training.

The continuation passes **212 relevant CPU tests**, script compilation and CLI-help checks. Independent checks also verified genuine adapter learning, exact unchanged pretrained/source fingerprints, selected-checkpoint hashes/steps and bounded artifacts. Measured allocations are sampled driver values; RSS overlaps with them and must not be added.


## Subsequent 97-material source refresh

The frozen continuation adds 28 complete materials and 84 native crops, producing **97 materials and 291 samples: 194 training and 97 validation**. All 291 pass decoded comparison with their exact source regions. The old 207 sample manifests, 882 cropped PNGs and index entries remain unchanged. Geometry verification covers 830 train/validation comparisons across all recorded map roles. The 415 required parent PNGs retain their original precision; every displacement is 16-bit. The prepared new families were not used in the completed four-material training trial.

AmbientCG Snow013/Snow014/Snow015 retain their original case-sensitive package/member filenames and provider URLs, separately from Poly Haven snow_01/snow_02. All 12 original Snow maps matched complete official ZIP members by SHA256 and byte count, with temporary archives removed. All new Poly Haven parent files matched the published originals. Four native grayscale-plus-alpha height maps are now supported without changing either component; training uses the recorded grayscale scalar alone under an explicit near-opaque-alpha policy.

At the user's request, the only 8-bit-displacement set, brick_4, was removed from the active source folder and moved intact to macOS Trash. Its six files and 118,313,574 bytes are checksum-recorded. It was not part of the historical index. Preparation now excludes 8-bit height while retaining other maps at their actual precision. One incomplete leaves_forest_ground source remains untouched and unindexed.

The locally retained source refresh records (`source-refresh-97/ingestion-97-summary.json`) preserve the actual audits and download identities. Two files named `.txt` were discovered to be complete byte-identical copies of recorded parent PNGs. The compact recipe restores these aliases by verified parent reference; no original note, parent or existing sample was changed. The historical 56- and 69-material recipes remain byte-identical, including their old embedded note payload.


The completed compact recipe is 319,073 bytes, with no embedded PNG payloads and no unresolved origin among its 415 parents. Exact restoration of both real PNG-as-TXT files and the unique README passed, including an idempotent second run; independent comparison confirmed unchanged index, all 291 metadata records and parent/source proofs against the retained full recipe. The final suite passes **218 CPU tests and 9 subtests** after the note-reference change. The locally retained current validation record (`continuation-validation-20261007/summary.json`) distinguishes these checks from earlier native/full-Python runs.
