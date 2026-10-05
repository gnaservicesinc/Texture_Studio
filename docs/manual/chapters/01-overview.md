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
