# IPDE Studio user manual

This manual describes the IPDE 0.9 development series. It is intended for technical users who understand files, numerical data and image coordinates but may be new to depth estimation or machine learning. The application is a pre-release: project, dataset, checkpoint and save formats may change before 1.0.0. Retain original photographs, calibration and a separate backup of important datasets and experiments.

## Choose the right application

| Application | Purpose | Recommended starting point |
| --- | --- | --- |
| IPDE Studio | Creates projects, stores their purpose and settings, and launches project applications. | Create a project with a meaningful name and location. |
| IPDE Extractor | Inventories and extracts original HEIF auxiliary arrays; optionally generates stereo estimates. | Inspect one representative photo before batch extraction. |
| Photo Studio | Examines HEIC planes, exports selected data, composes people mattes, and generates separately labelled depth derivatives. | Select the original depth or matte you need. |
| Raw Studio | Preserves RAW source files and available sensor or LinearRaw samples; produces separate developed RGB, depth and person masks. | Inspect the sample dtype and calibration before export. |
| Dataset Studio | Imports photos and datasets, generates teacher targets, reviews membership, assigns validation splits, and manages storage. | Review teacher targets before creating a long training run. |
| RAFT Studio / Trainer | Trains the experimental stereo-to-display student or an explicit stock RAFT experiment; validates, exports and compares models. | Compare the baseline and candidate on independent held-out captures. |

Projects share `project.ini`, a `workspace` and an `exports` folder. Dataset Studio owns dataset edits. Trainer reads datasets and links back to Dataset Studio when preparation or review is needed. A linked dataset remains at its original location; membership changes in Dataset Studio update that location. Removing a link removes the reference, not the dataset.

## Recommended workflow

1. Create a project and choose its purpose: effects/displacement, distance estimation, portraits/photo effects, or custom.
2. Inspect a typical input. Establish which camera grid, numerical representation and precision each useful plane has.
3. Extract original arrays before experimenting with derived products. Keep them alongside calibration.
4. For display-depth estimates and comparisons, import Spatial Photos in Dataset Studio. Teachers use the full display RGB photo. An ordinary portrait with an embedded depth map is not automatically a calibrated left/right stereo sample.
5. Generate teachers at a suitable processing resolution. Review fine structures, large flat regions and edges against the display photo. Inspect actual model-input dimensions; exclude unsuitable samples or individual teachers.
6. Compare direct display AI with existing stereo results before choosing a training experiment. The experimental stereo-to-display student learns full display targets from native stereo inputs; stock RAFT remains a separate native-left model.
7. Hold out independent scenes. Prepare a collection only when combining datasets or changing the split is useful; a compatible existing full-display dataset can be trained directly. Keep label units consistent within a run.
8. Begin with a short run. Inspect checkpoints and compare held-out outputs before investing in a long run, then export and explicitly select a chosen model. Keep direct display AI, experimental student output and stock stereo results identified.

The **Help** menu opens the offline interactive guide, this printable HTML manual, its editable Markdown source, GitHub and the bug form. **About** in every app provides version, build date, revision, build flags, compiler and Qt information; copy those details into a bug report.

# Numerical precision and output resolution

## Preservation is distinct from estimation

Original extraction preserves the decoded sample values, dtype and coordinate system. It does not apply gamma correction, tone mapping, normalization, sharpening or display scaling. A stored RGB code may still have an encoded transfer function: preserving it does not make it linear-light RGB. A numeric depth/disparity plane is treated as data.

A camera or encoder may already have discarded information. Float32 storage cannot restore missing levels in an 8-bit encoded map, and an Apple float16 decoded representation does not establish 16-bit scene measurement precision. A teacher estimate or interpolated map is a new derived product; it is not the original auxiliary array or a measured reference.

**Meters**, **inverse meters**, **pixel disparity**, **signed flow** and **normalized displacement** have different meanings. Depth in meters grows with distance. Disparity commonly grows as objects approach the cameras; its sign and principal-point correction depend on the defined coordinate convention. Use the supplied calibration and product metadata rather than guessing from whether a preview looks bright or dark. Normalized displacement is an explicit effects conversion; it must not replace raw or calibrated data.

Invalid samples can be `NaN`. Preserve them and decide how your destination renderer handles unsupported geometry. Replacing invalid values with zero can produce false troughs or infinite-distance surfaces. A dense RAFT estimate retains uncertain predictions; the separately named supported-depth output applies the stricter correspondence mask. Dense output and independently supported geometry are different products.

## Native stereo versus the display image

A Spatial Photo may contain a **2688 × 2016 left/right pair** and a separate **5712 × 4284 display image**. These are distinct camera grids. The output sizes are examples from these captures, not requirements imposed on every file.

Dataset teachers receive the **full display RGB photo**, not either stereo view. Relative-teacher metric anchors receive that same display photo. Full display-sized predictions are retained as lossless float32 EXR; retaining the model's native predictions is an optional Advanced storage setting. The manifest records native prediction dimensions in either case. A teacher output on the display grid therefore stays useful for direct inspection or effects work, even when it cannot supply a stock RAFT training target.

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

# Extraction, Photo Studio and Raw Studio

## Extract selected products

Import or drag a HEIC into Extractor and inspect its inventory. Select only the products you need. Raw depth, Apple native disparity/depth, HDR gain maps, portrait and semantic mattes, alpha planes, stereo views and display images depend on what that file actually contains. A missing auxiliary item cannot be reconstructed by extraction.

Raw extracted arrays stay in their stored coordinate system; recorded orientation does not imply that pixels have been rotated. Keep calibration with arrays when consuming them in another program. Optional NPY companions preserve exact sample representation. JSON manifests are optional and record inspection, calibration and provenance; important derived-product metadata is also stored in supported exchange files.

Enable RAFT or classical stereo only when estimating new geometry is your goal. RAFT requires the local source/runtime and a compatible checkpoint. Invalid explicitly selected model paths fail; model selection is not an implicit download. Reverse-consistency checks are additional computation. Classical estimates and their support masks use different evidence requirements; unsupported smooth surfaces are not automatically zero-height surfaces.

**Color Matching** is optional inference preprocessing. The selected Hero view remains the reference, and the other view's RGB distribution is matched. It does not edit original extracted arrays or provide a camera-profile calibration. **Tolerate camera detail differences** affects the classical matcher's inference copies. These controls are unnecessary when exporting only raw auxiliary planes.

## Photo Studio

Select a file, then an operation and product. **Original** exports the chosen plane. **Depth upscale** creates a separately labelled RGB-guided interpolation on a supported registered grid. **Learned depth** and **RAFT depth** create estimates. Inspect original metadata to establish which units and camera reference you selected.

For a people cutout, select the portrait and useful semantic layers. The composite uses their maximum coverage, which avoids artificially increasing opacity at overlaps. Check hair, skin, glasses and fine edges in the preview; deselect incorrect layers. Sky is not a people matte. **Isolate** exports a selected region. Color values stay separate from the new alpha computation. **Export & copy** copies exact PNG bytes where appropriate and file URLs for other formats.

HEIC sharing/repair is supported only when the available writer can preserve every required retained image, auxiliary and calibration item. A refused rewrite leaves the original intact. Privacy cleanup addresses supported recorded metadata; recognizable people and places remain visible in the picture. When a container cannot be repaired exactly, export separate lossless arrays for your workflow.

## Raw Studio

Use **Original** to preserve the source RAW file. Available sensor/LinearRaw exports retain the stored raster, source dtype and margins together with camera calibration. Some Apple ProRAW files contain demosaiced LinearRaw samples rather than a color-filter mosaic; do not assume every RAW file has the same sensor representation.

**Developed RGB** is a separate derived product. The native extended-linear RGB result can contain negative values and values above one. These are retained numerically. Teacher inference uses a separate model input copy. **Person mask** uses Apple Vision on a viewing copy of the corresponding developed grid. **Learned depth** generates a separate estimate.

The optional region is expressed in the selected product's pixel coordinates. Sensor coordinates, developed RGB coordinates and a depth model's native grid are not interchangeable. Preserve full sensor calibration before cropping or converting data externally. A preview is a convenience display, not the scientific output.

# Datasets, teachers and validation splits

## Build and review

Dataset Studio discovers compatible calibrated Spatial Photos, generates one to three teacher targets, records skipped files, and keeps existing arrays separate from previews. A teacher is a local depth model that creates estimated labels. Select a model/checkpoint, its processing size and device. New datasets use the **full display RGB photo** as the teacher reference. The metric-anchor model also uses that display photo. Teachers do not run on stereo-left or stereo-right as a fallback. The full display-sized target is retained with its grid and provenance. Native teacher predictions and confidence arrays are optional intermediates, omitted by default.

DepthPro produces an estimated metric scale. Depth Anything V2 and the current DA3 configuration produce relative depth. **Relative teacher scale** can add an explicitly anchored estimated-meter map when meter scale is needed. DepthPro anchoring fits estimates made from the same full display photo; it is not measured camera calibration or proof of true distances. Relative maps remain useful for visual review/effects without that conversion. Inspect anchored results, and do not add an unnecessary second anchor to an already metric teacher. Anchoring does not by itself solve the display/stereo grid difference.

The experimental **stereo-to-display student** uses these full display targets directly. Targets are not resized or registered into stereo-left coordinates; stereo RGB remains at its native dimensions. Each selected label must match its display RGB provenance and exact H×W grid. Positive finite pixels, restricted by any recorded validity mask, contribute to training. Targets retain their stated units: a run must use one convention throughout, either meters, relative depth or relative inverse depth. Separate incompatible teacher-unit runs or choose an explicitly anchored meter-scale target; no silent reciprocal/scale conversion is applied.

Stock RAFT is a separate native-left experiment. Display-to-stereo alignment can vary with scene distance when camera viewpoints differ; a fixed warp does not generally solve that geometry. Native-left teachers, registered display-to-left targets and left-grid measured references are excluded from the display student. An independently measured **display-grid** reference can support supervised display training when its grid/provenance is declared correctly. Older native-left datasets need regenerated full display teachers for this architecture.

In **Photos and depth**, inspect the full display teacher beside its display RGB reference. If reviewing an explicit native-stereo experiment, inspect its left-grid target beside stereo-left RGB and check its stated provenance. Use native pixels, overlays and the lit surface preview. Review large flat surfaces, silhouettes, thin features, repeated textures and reflective areas. Remove a bad teacher target while retaining useful variants, or exclude the entire photo. Removed entries remain restorable. Membership and split changes save in the manifest; source arrays are preserved.

**Add photos** registers new captures and then generates missing targets. **Add from dataset** reuses prepared arrays. **Link dataset** references an existing directory; keep that directory available. **Compact** writes a new verified lossless EXR/PNG copy and deduplicates identical arrays. It preserves all existing scientific planes, including any intermediate or auxiliary planes in older datasets. Archive/cleanup is a separate storage action; verify paths and backups before deletion.

## Storage and size planning

New datasets store each distinct floating depth result in a **single-channel lossless ZIP OpenEXR**. Teacher float32 values remain float32, with no alpha, RGB duplication, gamma correction, normalization or resampling. Stereo RGB and the display RGB used for review use exact unsigned 8/16-bit PNG. Calibration, units, coordinate grids, source hashes, teacher provenance and split assignments remain in `dataset.json`. Masks or arrays whose dtype/layout cannot be represented exactly by these image formats use lossless NPZ. Writers decode and verify the complete array bit pattern before accepting a file, including NaN payloads and signed zero.

One teacher with metric anchoring disabled normally needs one full display-depth EXR per photo, plus the stereo pair and display RGB. Equivalent positive-finite validity masks are derived from the unchanged depth values instead of stored separately. Masks with additional rejection rules remain separate. New datasets omit unrelated embedded auxiliaries; original HEIC files and the extraction workflow still preserve them. Enable **Keep teacher intermediate arrays** or **Copy all embedded auxiliary images** in Advanced to retain those additional payloads. The CLI equivalents are `--retain-intermediates` and `--preserve-auxiliary-assets`; `--array-format numpy` uses NPZ and `--uncompressed` uses NPY.

Multiple teachers, a DepthPro scale estimate and an anchored result represent different numerical maps and require additional files. The detail preset leaves metric anchoring off: relative depth is sufficient for the default display student when all targets use one unit convention. Enable anchoring when an estimated meter scale is useful, and review its errors.

A 5712 × 4284 float32 depth plane contains 97,880,832 bytes (93.35 MiB) before compression. Three hundred such planes total about 27.35 GiB before compression; RGB and any extra labels add to that total. Lossless compression depends on the values and cannot promise a fixed size. On one existing full-size teacher map, NPZ used 77.5 MiB and ZIP EXR used 65.6 MiB with bit-exact verification. This is a sample measurement, not a guaranteed ratio. Removing unused variants and avoiding copied datasets usually matters more than the container alone. Shared collections and linked datasets avoid repeating existing payloads.

## What independence means

Validation tests whether a model works on examples withheld from training. Use different scenes or captures, not alternate teachers of the same photo. Category names such as "rooms" are organizational labels. They do not establish scene independence.

An **independent scenes** declaration records your knowledge of scene groups; it does not verify the photos automatically. Keep related burst/capture/scene components on one side of the split. Known duplicate identities and teacher variants are protected, but unknown related photographs can still leak into validation. A low score on leaked validation data is misleading.

Choose **Global random**, **Equal per dataset**, or **Separate validation datasets** according to the experiment. The seed makes the split repeatable. Percentage applies to independent components, so the number of teacher-entry samples can differ. Training requires eligible training and validation examples. A 0% or 100% split normally cannot provide both.

Training directly from a suitable dataset avoids unnecessary collection preparation. Prepare a collection to combine datasets, change splitting strategy or make a portable reviewed snapshot. Same-filesystem shared storage avoids redundant copies; use an explicit copied output for migration across volumes. Inspect the manifest and keep source hashes/provenance.

## Integrity checking

Full verification checks array content, expected hashes, shapes, calibration and known split overlap. It can be costly on large compressed datasets because arrays must be read and decoded. For supported IPDE-native datasets, training checks metadata and the split at startup, then verifies each RGB, target and mask array's checksum and geometry when that sample is first consumed. Generated native sets and their app-composed, curated, compacted or edited descendants qualify when every retained sample preserves its native stereo and calibration provenance. This avoids rereading the entire payload before the first update, including for supported existing native datasets. Corruption aborts training instead of silently dropping a sample.

External or unknown datasets receive a full scientific-array preflight. Training does not write a trust receipt into an input dataset, and no folder name exempts data from checking. Deferred integrity checks do not prove labels are accurate, independent or appropriate for your task. Review the actual targets. Dataset fingerprints and hashes establish identity and detectable change, not truthful measurements.

External imports require explicit left/right arrays, registered meter-depth, calibration and source/teacher identity mappings. Generic pictures, screen-rendered depth or unanchored relative estimates cannot be substituted for calibrated stereo labels. Optional Hugging Face import requires its dependency to be installed and uses data-only loading. See the [official Datasets loading guide](https://huggingface.co/docs/datasets/loading).

# Training controls and model selection

## Choose the output task first

**Experimental stereo → display depth** is the default student. It receives both native stereo RGB images, encodes them whole, and predicts directly on the full display grid. It learns from the original, unregistered display teacher target. Every supported positive finite target pixel contributes, in bounded decoder tiles. See [Experimental stereo-to-display model](#experimental-stereo-to-display-model) for its architecture and limits.

**Stock RAFT native-left disparity** remains an explicit separate experiment. It learns calibrated left-grid flow/disparity from native stereo crops. A full display teacher cannot directly supervise its existing decoder. Selecting a student changes dataset eligibility, training counts and error units; check the displayed plan before starting. Older left-grid teacher datasets need regenerated full-display targets for the display student.

## Distillation, supervised and mixed modes

**Distillation** trains a student to agree with teacher-generated targets. It can also learn systematic teacher mistakes, smoothness and incorrect scale. More training on blurry targets can strengthen the blur. A lower validation error measures agreement with withheld teacher labels, not independent physical accuracy.

**Supervised** training uses reference targets identified as measured/calibrated labels. For the display student, those references must belong to the full display grid. For stock RAFT, they must match the native left camera and support physical stereo geometry. Their source, precision, alignment and validity matter more than the mode name. Do not relabel teacher predictions as measured references. **Mixed** uses eligible reference targets where present and compatible teacher targets elsewhere. **Auto** chooses from eligible label types. Reports identify which targets contribute.

The display student retains one unit convention per run: meters, relative depth or relative inverse depth. Relative targets do not require a metric anchor. An explicit DepthPro anchor can supply estimated meter scale when useful, but still supplies an estimate. Stock physical stereo-flow labels require compatible meter scale and calibration. For effects, compare direct display AI, the trained display student and separately identified stock stereo results. For physical accuracy, use independently measured held-out references.

## Epochs, Total Steps and Steps Per Update

An **epoch** traverses every eligible image/teacher entry once in shuffled order. Each display entry uses its whole supported target in tiles; each stock RAFT entry draws a native-pixel crop. Different teacher variants are separate entries. **Steps Per Update** accumulates that many image gradients before one optimizer update; it defaults to one. **Total Steps** counts optimizer updates rather than image visits or decoder tiles. Increasing accumulation reduces updates per epoch and changes the optimization schedule; it does not reduce work required to visit every entry.

For `T` eligible train entries and accumulation `A`, updates per epoch are `ceil(T / A)`. An epoch limit `E` plans `E × ceil(T / A)` Total Steps. The partial accumulation at an epoch's end is flushed. Example: 101 entries, 4 Steps Per Update and 3 epochs produce 26 updates per epoch and 78 Total Steps.

Choose **number of training epochs** for equal image exposure; it stops after that many complete epochs. Choose **total number of training steps** for an exact update budget; it can finish partway through an epoch. Progress shows epoch image/tile activity and cumulative optimizer steps. Filtering can make the eligible count smaller than the library's visible entry count.

## Quality, length and advanced settings

With advanced settings hidden, **Quality** adjusts decoder tile size, added display-student feature/decoder width and RAFT refinement iterations. Native RGB and target dimensions remain unchanged. For stock RAFT, quality instead adjusts native crop size and refinement iterations. Higher quality costs time or memory; it is not an accuracy guarantee. **Length** sets a dataset-aware duration and validation error goal. Inspect the plan: defaults cannot judge label quality or prove that full-native encoding fits memory.

Low/medium/high settings use 256/512/768 tile pixels, added feature widths 16/24/32 and 8/16/24 RAFT iterations. In stock mode, those pixel values specify native crops. Fast/medium/slow starts from 1,000/5,000/15,000 update budgets with 5/20/50 maximum epochs, capped according to eligible dataset size. Stock full-network scope is selected only with at least 500 entries and medium/high quality. Display auto-full additionally requires recorded native inputs no larger than 512 × 512 pixels; full-size or unknown native inputs keep RAFT frozen and train the added heads. Advanced full-network display training remains an explicit memory experiment. Display early-stop goals are 0.20/0.10/0.05 fractional error; stock goals are 2/1/0.5 pixel flow MAE. Each requires full-validation confirmation. Validation uses 16 random entries for at least 100 training entries, otherwise the full eligible set. These are adjustable development defaults.

The display student's limited scope trains its added encoders/query decoder while RAFT stays frozen. Full-network scope adapts RAFT too and uses more memory. Stock limited scope updates RAFT's refinement block. Advanced settings expose scope, mode, tile/crop size, iterations, learning rate, validation, checkpoints, accumulation and end conditions. Initial display-head learning rate is `1e-4`; stock fine-tuning starts at `1e-5`. Changing tile size mainly changes decoder memory and dispatch overhead; it does not add scene context by cropping the inputs.

## Validation cadence, units and sample limit

Validate **every epoch** for frequent feedback or **when saving a checkpoint** to reduce overhead. End-only checkpointing with checkpoint-only validation gives little intermediate feedback. A sample limit of zero uses all eligible validation entries. Positive `N` selects up to `N` entries randomly each pass; subsets can make scores fluctuate. Record the policy and sample count when comparing runs.

Display validation reports `mean(abs(prediction - target) / target)` over supported display pixels. A fraction of 0.10 is a 10% average absolute relative difference, not a pointwise bound or calibrated confidence. Raw mean absolute error is also reported in the declared target units. Stock RAFT reports stereo-flow MAE in **pixels**. Compare like architectures, units, targets, splits and policies. Falling training error with worsening independent validation can indicate overfitting.

## Checkpoints, stopping and resuming

Save intermediates every epoch, every `N` epochs, every `N` optimizer steps, or only at the end. **Save checkpoint now** queues a save at the next completed safe update. It appears in the model library and can be exported while training continues. Saves consume disk space and pause work briefly. Exported display models are named `display-model.pth`; stock exports remain separate RAFT artifacts. The checkpoint/report records architecture and units.

An **early end error** threshold uses display fractional error or stock pixel MAE according to the chosen student. If a random subset passes, Trainer reruns full validation and stops only if the full result also passes. Choose goals from task requirements and baseline behavior, not solely to end a run sooner.

**Stop** saves an orderly checkpoint at a safe update boundary. Resume state includes optimizer state, current shuffled order/cursor and random-generator state. Keep original run files and dataset identity. Resume into a new checkpoint destination using the same manifest and compatible architecture, units, mode, scope, tile/crop size, iterations, accumulation, learning rate, seed and support policy. An inference-only exported model is different from full resumable state. Force quit, terminated processes, power loss or disk failure cannot guarantee a final save.

Nonfinite error, exactly zero error or error above **Maximum safe training error** stops and records a reason. Display mode uses mean fractional target error; stock mode uses weighted mean absolute flow error in pixels across refinement iterations, normalized independently of iteration count. The optimization objective is recorded separately. Failure reports/checkpoints preserve recoverable finite state, not a guarantee of a usable model. Inspect the reason, targets and settings before resuming.

## Compare and deploy

Review held-out full images, fine features, silhouettes, flat surfaces, reflections and unfamiliar captures. Display and stock models belong to different grids: compare each map with its own RGB reference rather than calculating an invalid pixelwise difference between them. Direct display AI provides a useful separate baseline. Export and explicitly select a chosen checkpoint in the project. The [official RAFT-Stereo repository](https://github.com/princeton-vl/RAFT-Stereo) describes upstream correspondence models; this app's display student is a separate experimental architecture. A brief run or a falling teacher-agreement score does not establish real-world accuracy.

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
| Blurry full-sized output | Teacher processing size, native predictions, student architecture and labels. | Inspect source/target at native pixels; increase appropriate teacher detail and review before retraining. |
| Tiny "patch pixels" | Chosen student: display-query tile versus stock native crop. | Display tiles visit the full target; stock crops preserve original pixel scale. Neither squeezes the whole photo into the tile. |
| Dataset checking dominates startup | Dataset origin/schema and external import. | Supported native datasets check payloads as consumed; external/unknown datasets require full preflight. |
| Training is on CPU | Reported device and available PyTorch backend. | Select MPS on compatible Apple Silicon; inspect the failure if it is unavailable. |
| Low GPU load between updates | Decompression, preprocessing, synchronization or frequent validation. | Inspect phase timings, use suitable cache/worker settings and reduce validation overhead. |
| Validation error varies | Different random subsets or scarce independent scenes. | Increase sample count or use full validation; compare identical policies. |
| Validation stays good but new scenes fail | Scene leakage or insufficient diversity. | Rebuild an independent split and add genuinely new scenes. |
| Error diverges / zero / NaN | Incorrect calibration, invalid targets, loss or runtime problem. | Read the stop reason; review data before resuming the saved state. |
| Display target is ineligible for stock RAFT | Display and native-left grids represent different outputs/viewpoints. | Use the experimental display student for that output task, or compare direct display AI; do not substitute a left teacher or force a registration/resize. |
| EXR looks black in an image viewer | Numerical units, range or NaN handling. | Inspect the values in a data-aware program; use a separate visualization. |
| HEIC rewrite refused | Writer changes retained data or loses auxiliary/calibration items. | Export separate exact arrays; retain the original container. |
| A linked dataset disappears | Its original directory moved or became unavailable. | Restore the source or update the link in Dataset Studio. |

## Install and report issues

macOS releases are pre-releases until 1.0.0 and are not notarized. The finished IPDE Studio bundle contains its project apps and deployed Qt libraries. Source developers can run `make build` and `make install`; installation requires the existing app and its subapps to be closed. `make DESTDIR=/path/to/staging/dir install` stages the application under that root's Applications folder. Keep project data outside the application bundle.

Use **Help → Report Bug** to open the [issue form](https://github.com/gnaservicesinc/ipde/issues/new). Include copied About information, app role, input format/grid sizes, selected products or training settings, steps to reproduce, and the relevant error/report. Share a minimal example when practical; a model can require substantial runtime resources to reproduce. Include whether a dataset is local, linked or imported and whether validation used all images or a subset.

The [IPDE repository](https://github.com/gnaservicesinc/ipde) contains source, releases and development documentation. [pillow-heif documentation](https://pillow-heif.readthedocs.io/en/latest/) covers HEIF decoding. This manual's recommendations are starting points for experiments, not claims of measured accuracy.

# Experimental stereo-to-display model

## What this model learns

The experimental student addresses a different output task from stock RAFT-Stereo. For a capture with 2688 × 2016 left/right views and a 5712 × 4284 display image, the teacher receives the complete display RGB photo and makes a target on that display grid. The student receives only the native stereo pair and learns to predict the same display grid. Display RGB is not a student inference input. The target is not homography-warped into the left view, and a stereo-left teacher is not substituted.

Its checkpoint schema is `ipde-display-depth-v1` and architecture is `stereo-display-query-v1`. This is a new experimental model, not an upstream RAFT checkpoint with an output-size option. Stock RAFT's native correspondence/depth/disparity products remain separately available.

## Encoder and query decoder

RAFT first supplies native-grid stereo correspondence context. Added convolutional encoders extract features from each native RGB view and from that flow. The model also pools a global scene descriptor. These are learned internal feature representations; they do not change the stored RGB or camera arrays.

For every output display pixel, the decoder receives its normalized coordinates, the input/output size ratio, sampled stereo features and global context. It learns separate sampling offsets into left/right features and predicts a positive scalar value. These offsets can vary by pixel and scene; they are not an imposed camera homography or a transport of teacher values. The loss is evaluated against the original target at that display pixel.

Decoder queries are evaluated in bounded tiles and assembled into the full float32 map. Native stereo inputs are encoded whole. Tiling limits temporary decoder memory; it does not squeeze the full photo into a tile-sized image. An epoch visits every eligible image/teacher entry, including all of its supported positive finite display pixels. Larger query tiles mainly change memory/dispatch overhead. Output tiling cannot guarantee that full native RAFT encoding fits a chosen device.

With the limited update scope, RAFT stays frozen while the added stereo encoders and query decoder learn. Full-network scope also adapts RAFT and costs more memory. The initial output head is new: the display model needs training before its predictions are useful, even though its RAFT component starts from a pretrained checkpoint.

## Target units and error scores

Each run retains one target convention: `meters`, `relative_depth` or `relative_inverse_depth`. The model does not use the stock `focal × baseline / disparity` formula to turn its output into meters. Meter-labelled output is learned agreement with meter-labelled targets; teacher estimates or model anchors still do not establish measurement accuracy. A relative or inverse-depth checkpoint stays identified as such.

The optimization objective, validation end condition and divergence guard use mean absolute **fractional** error: `mean(abs(prediction - target) / target)` over valid display samples. A value of 0.10 corresponds to an average absolute relative difference of 10%; it is not a maximum pointwise error, a calibrated uncertainty or proof of physical accuracy. Reports also give raw mean absolute depth error in the target's units. Stock native RAFT experiments instead report stereo-flow MAE in pixels. Compare scores only within the same architecture, unit convention, target/split and validation policy.

Training reports use `ipde-display-training-report-v1`; resumable state uses `ipde-display-resume-v1`. Keep the original manifest and compatible mode, scope, query tile size, RAFT iterations, accumulation, learning rate, seed and model/unit configuration when resuming into a new destination. An exported inference checkpoint remains a different artifact from full training/resume state.

## What full-sized output does and does not establish

The decoder evaluates every requested display pixel instead of simply enlarging a completed low-resolution depth map. That satisfies the output-grid contract. It does not create independent camera measurements or guarantee accurate fine structure. A high-resolution teacher can contain plausible details, smoothness, incorrect scale or other mistakes. The student has less visual information than a teacher receiving the larger display RGB, and unseen areas/viewpoint changes can remain ambiguous.

Compare independent held-out captures at native display pixels. Inspect thin objects, silhouettes, flat surfaces, reflections, holes and unfamiliar camera/framing conditions. Compare against direct display AI and clearly identified stock stereo results. A short test, a falling teacher-agreement score or a visually detailed map is not evidence of measured accuracy. Retain the original teacher/native arrays and experiment provenance while assessing whether this model serves an effects goal.
