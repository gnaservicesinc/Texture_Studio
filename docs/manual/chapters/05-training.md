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
