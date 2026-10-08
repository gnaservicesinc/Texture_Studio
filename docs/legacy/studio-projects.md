# IPDE Studio projects

Create a project in one dialog: name, main purpose, and parent folder. The default parent is your macOS Documents folder. Studio stores the purpose in `project.ini`; the purpose remains visible and editable when the project is open, and the Extractor and Trainer use it to choose their product and teacher presets.

- **Effects / displacement maps:** start by building or linking a dataset, review the labels, then assemble and train. Compare the trained checkpoint with the generic RAFT baseline on held-out captures before explicitly selecting it. Extract maps also works with the generic model while the dataset is being prepared.
- **Depth estimation:** inspect capture calibration and coordinate registration, then extract supported depth. A model producing meter units still needs independent measured references to establish physical accuracy. Raw Studio supports sensor-source workflows.
- **Portraits / photo effects:** use Photo Studio to inspect person mattes, choose mask layers, and export cutouts or derived depth products. Original auxiliary arrays remain available in Extract maps. Portrait mattes apply to people; ordinary portraits do not supply a calibrated RAFT stereo pair.
- **Custom / manual:** every app remains available. Establish the required output arrays, coordinate grids, units and precision before choosing inference or training options.

The project panel shows dataset, sample, source-photo and trained-model counts, along with unavailable links, incomplete datasets and missing selected checkpoints. Its suggested action changes as datasets and training reports appear. Selecting a model remains an explicit action in Trainer. **Show all apps** exposes other tools without changing the main purpose.

The **Datasets** button opens Dataset Studio with its own application icon. **Trainer** opens RAFT Studio. Both share the project's dataset library, linked sources, purpose and selected RAFT checkpoint. Dataset Studio owns imports, links, review, reviewed copies, training-set preparation, compacting, archiving and dataset cleanup. Trainer reads selected datasets and owns training, model comparison, export and run cleanup. Links to another step ask the project hub to launch the correct app or focus its existing window, retaining the requested dataset and step.

## Reusing datasets without copying

In Dataset Studio, **Link datasets from project…** lists datasets from another recent project, or a project selected with **Other project…**. Select the datasets to reference. **Link dataset…** accepts a standalone folder containing a valid IPDE `dataset.json`.

Studio stores each canonical dataset directory in the project's `dataset_links` list. Dataset Manager and Trainer read those original paths. Curation, composition and compacting save new outputs under the current project's workspace; archiving is limited to datasets owned by that workspace. Linked source folders must remain available. Moving or deleting a source produces an alert. **Manage links…** removes references without deleting files.

Dataset review and training checks help detect accidental mistakes in labels, scene grouping and provenance. They do not assess external datasets for posing, copyright, inappropriate content or deliberate misrepresentation. Future content-review tooling belongs in a separate explicit review stage; a linked dataset is not thereby certified.

## Menu bar and sessions

On macOS, Studio stays available in the system menu bar after its main window closes. The menu can reopen Studio, change projects, create or open projects, and launch the project's apps. Each app role has one session per project. Distinct projects can keep their own apps open at the same time.

Use **Quit IPDE Studio…** to end Studio. If project apps are active, the menu asks whether to close them too, because Studio owns their shared session. On platforms without a system tray, keep Studio open until its project apps have closed.

## Persistence

Existing `ipde-project-v1` projects remain compatible. The small INI file records:

```ini
[General]
schema=ipde-project-v1
name=My project
goal=effect/map
```

The supported `goal` values are `effect/map`, `depth-estimation`, `photo-effects` and `manual`. `dataset_links` is written as a Qt `QStringList`; use Dataset Studio's link controls rather than hand-editing its escaping. Shared RAFT choices use `raft/root`, `raft/model` and `raft/member`. Project data remains in `workspace/datasets`, training runs in `workspace/runs`, and default extraction exports in `exports`.

The project **File processing threads** override is stored in `performance/workers`. Automatic (0) uses available CPU cores; an explicit positive count limits CPU file and preparation tasks. Existing projects default to automatic. Dataset Studio and Trainer read the same setting when starting each task. Full-image jobs also cap concurrency to bound decoded memory. GPU device controls remain separate: model inference/training can use MPS, while file reads, hashing and lossless compression use CPU workers.

Purpose changes choose presets in the tools; they do not alter original source arrays or silently select trained checkpoints. Display previews and derived resized maps must remain distinct from raw preserved data.
