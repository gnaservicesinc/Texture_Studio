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
