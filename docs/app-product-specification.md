# Texture Studio: product specification

Build a standalone macOS application for Apple Silicon that turns photographed surfaces into editable materials for Blender, manages paired material-map datasets, refines material models, and lets the user inspect and compare the results. This specification is self-contained. It describes required behavior and deliverables; it does not prescribe an architecture, programming language, framework, or implementation technique.

## Workspaces and navigation

- Provide a photo-to-material workspace, Dataset Studio, Model Training, Checkpoint Compare, Material Review, and a saved-model library. These tools must be reachable from the application and usable in separate windows.
- Keep the active dataset, selected subject, crop, map, model, and current operation identifiable. Navigation must not silently change the dataset, model, or selected resolution.
- Provide keyboard navigation, accessible names, descriptive tooltips for icon actions, resizable inspection areas, and direct typing or pasting into numeric controls. State units and allowed ranges. Invalid entries must not corrupt saved settings or start an invalid operation.
- Remember user settings, recent datasets, relevant view preferences, and export locations across launches. Saved photo recipes must restore their source photo, adjustments, material settings, and model selection.

## Photo preparation and material generation

- Import HEIC/HEIF, JPEG, PNG, TIFF, and supported camera RAW photographs. Show the source dimensions, available camera information, and relevant auxiliary-image information. Support opening files through the application and dragging supported files into it.
- Provide perspective correction, horizontal and vertical tilt, rotation, lens correction, focal-length override, square crop size and position, and a reset action. Keep the chosen crop inside valid source pixels.
- Prepare a diffuse map with adjustable broad illumination balance while retaining fine surface detail. Allow available HDR gain-map information and registered companion photographs to contribute. Clearly identify unavailable information or estimated calibration.
- Let the user choose flat relief, an attached registered surface-height map, or a compatible refined material checkpoint. Camera portrait depth must not automatically become surface displacement. Brightness-derived artistic relief must be optional and off by default.
- Produce diffuse, roughness, tangent-space normal, and displacement maps. Provide controls for relief strength, inversion, slope removal, artifact cleanup, near-flat-surface protection, roughness, material width, and displacement scale. Express physical dimensions in meters.
- Model inference must use the same prepared diffuse pixels, geometry, and crop that the user reviews and exports. Mark results stale when relevant settings change. Generating, inspecting, and exporting an unchanged material must reuse its completed result.
- Offer material output sizes of 1024, 2048, 4096, and 8192 pixels independently of training size. Larger output dimensions must not be presented as recovering absent source detail.
- Preview each available map and provide full-quality inspection. Display adjustments must not alter numerical map values or source files.

## Material export and original auxiliary data

- Export all four material maps into a user-selected material folder: diffuse as 16-bit sRGB PNG; roughness, normal, and displacement as selectable 16-bit or 32-bit floating-point EXR.
- Numeric maps must retain their numerical meaning without photographic gamma, tone mapping, or automatic range stretching. Float32 export must preserve the current Float32 sample precision.
- Export Blender setup instructions and a material manifest identifying map files, color interpretation, OpenGL +Y normal convention, material width, and displacement scale. Support normal shading and geometric displacement without applying the same relief twice.
- Export untouched original map files independently of display zoom or contrast. Opening a map in an external editor must use an editable copy and preserve the original.
- Provide a separate original-auxiliary-data export for information exposed by the photograph reader. Include untouched auxiliary buffers, descriptions of their dimensions and layout, relevant metadata, and exact scalar-array exports where supported. State the reader's coverage limitations. Preserve the original photograph.

## Dataset creation, import, and curation

- Create, name, reopen, rename, describe, and close datasets. Show recent datasets and their locations. Let the user choose the dataset location, working folder, native resolution, and validation settings.
- Register original diffuse maps with the available displacement, roughness, and normal targets. Support multiple registered diffuse color variants for the same geometry. Keep distinct source resolutions and source families identifiable.
- Import a folder of paired material maps. Recognize common diffuse/albedo/color, displacement/height, roughness, OpenGL/DirectX normal, numbered-color, and provider-resolution filenames. Show the proposed materials, available targets, omissions, and conflicts before committing the import.
- Do not fabricate missing targets, reinterpret bump as displacement, or claim that low-precision height is high-precision data. Explain which runs each source set can support. Omit unrelated maps such as specular or ambient occlusion from runs that do not use them.
- Verify actual dimensions and available channels. Paired maps and color variants must share registered geometry. Ambiguous role assignments, incompatible dimensions, unsupported transparency, and unsuitable target precision must be reported clearly.
- Preserve original map bytes, dimensions, precision, color meaning, and normal convention. Registration must not duplicate, resize, pad, denoise, or color-convert original images.
- Browse subjects, their source sets/crops, and their maps. Search by subject/material/sample identity. Show source dimensions, precision, selected native grid, map identity, and training eligibility alongside inspection.
- Allow approve, exclude, reapprove, mark unreviewed, and saved notes. Apply crop review status consistently to its paired maps. Keep reviews tied to the actual source and selected grid.
- Remove material membership while retaining originals. Dataset deletion must move only app-owned dataset data to Trash, preserve original sources and unrelated files, and allow recovery. Missing-source handling must distinguish an absent file from a temporary read failure.

## Required folder-selection behavior

- Every subject folder and nested crop/source-set folder must have a small white disclosure arrow that is easy to click. Activating the arrow must select that folder, visibly highlight it, and reveal or collapse its children as appropriate.
- Selecting a subject folder must show an available image from that subject in the adjacent inspection pane. Selecting a nested folder or individual map must update that pane to the corresponding image. A successful selection must never leave an unrelated old image or an empty pane when a usable map exists.
- Show the map icons and applicable inspection/review actions below the image. The user must be able to select each available map and immediately see which one is active.
- Folder selection, expansion, map selection, review controls, validation checkboxes, scrolling, searching, and keyboard navigation must work together without losing selection, triggering unrelated actions, or crashing.
- Loading indicators and empty/unavailable states must explain the current state. Unsupported or absent maps must not appear as usable controls.

## Native training grids and preparation

- Offer training grids of 256, 512, 1024, and 2048 pixels, filtered by available source pixels. Every input, target, and training-review image must use the explicitly chosen grid. Never silently lower that grid or use a hidden smaller training crop.
- Exact-size source images must be usable directly. Larger sources must yield exact registered crops: one centered crop for ordinary sources; top-left, top-right, and bottom-right crops when both source dimensions are at least 8192 pixels. Use matching regions across roles and diffuse variants.
- Smaller source sets must remain inspectable and be marked unavailable for that run. Never resize or pad them to manufacture eligibility.
- Preserve source integer codes and numeric target ranges. Account explicitly for diffuse color encoding and OpenGL/DirectX normal convention. Do not apply photographic correction or per-crop normalization to numeric training targets.
- Prepare only the input and target roles required by the selected operation. A diffuse-only operation must not prepare normal, roughness, or displacement images. A displacement run needs diffuse and displacement; a roughness run needs diffuse and roughness; a normal run needs diffuse and normal. Existing unrelated dataset maps may remain available for inspection without becoming prepared samples or training inputs.
- The sample count must represent useful samples for the requested run. Unused maps must not inflate counts, disk use, loading time, or preparation time.
- Dataset indexes and review records must remain small relative to the image data. Do not embed image or pixel buffers as Base64 text. Do not store invented values or repeated facts already reliably available from the source files, filenames, or selected dataset settings. Integrity records must have a clear purpose; redundant full-file digests and repeated unchanged-file scans must not add unnecessary work.
- Reopening or reusing an unchanged dataset must not repeat completed preparation. Changing the grid or target must invalidate only results that are no longer applicable. Temporary prepared images must be removable when the run stops or finishes, without removing original sources or saved reviews.

## Performance, resources, and operation control

- Keep the application responsive during import, preparation, model download, inference, training, and export. Report meaningful progress, the active phase, completed work, and any resource-related limitation. Provide working cancellation.
- Preparation must not attempt to consume all available RAM, cause runaway swap, or crash the machine. Large datasets must be usable without memory growth proportional to the entire dataset. Resource use must remain bounded on the actual machine for the duration of each operation.
- Expose the preparation-worker setting in Settings. Its initial default must equal the CPU cores available to the application on that machine. Persist explicit user changes. Report the effective worker count if memory or current machine conditions require less concurrency.
- Preparation must avoid unnecessary image copies, unrelated map decoding, duplicated temporary images, and repeated serialization. For an unchanged exact-size dataset, its work must be limited to the necessary registration and integrity checks rather than processing every pixel again.
- Cancellation must finish outstanding owned writes before cleanup and leave the dataset and previously saved checkpoints usable. Interrupted work must not publish partial files as completed results.
- Concurrent windows must not overwrite each other's dataset changes, silently use stale import proposals, or remove files used by an active run. Source changes must invalidate affected review/preparation results rather than silently reusing them.

## Model refinement and validation

- Refine a compatible pretrained material model for displacement, roughness, or normals using the selected whole dataset or selected material. Identify the dataset, material scope, target, size, and starting model before training starts.
- Support starting from a base model or compatible saved checkpoint. Reject incompatible base identity, target, trained modules, or model layout with a clear explanation.
- Expose updates per map, time limit, quick-validation interval, checkpoint interval, and relevant developer refinement controls. Provide progress, elapsed work, validation results, and a usable operation log.
- Support Stop, Save Checkpoint Now, and Stop and Save. Saving must occur at a complete update boundary. Save Checkpoint Now must validate and save while allowing training to continue. Stop and Save must leave a usable saved result. Already saved checkpoints must survive an aborted run.
- Let the user enable validation, select eligible subject folders, set a maximum subject percentage, set a maximum validation-crop count, and set the quick-check count. All target plans must share the same selected subject regions where those targets exist.
- Enforce percentage/count limits, with percentage caps rounded down. Prefer a different available source region for a known-subject learning check. Clearly label possible overlap and distinguish these checks from evaluation on unseen materials. Do not claim independent validation when none exists.
- Checkpoint saves and final exports must evaluate the full configured validation pool. Loss values must not automatically approve or promote a model.

## Comparison and detailed review

- Compare reference maps, the unrefined base, and one or more saved checkpoints using identical prepared diffuse pixels at the chosen model grid. Also accept a fresh test photograph prepared through the same photo-to-diffuse workflow.
- Retain each pane's subject, map, model/checkpoint, source dimensions, and displayed-grid identity. Keep the source reference visible while the user chooses which additional predictions to display.
- Provide linked pan and zoom, Fit, 100%, 200%, zoom buttons, direct zoom entry, and pop-out comparison windows. Numeric display contrast and midpoint controls must affect only the display and must be resettable.
- Allow full-source inspection separately from crop/model-grid inspection. Never imply that opening the original source runs a full-source model prediction.
- Support individual and grouped original-map export, editable-copy opening in GIMP, and opening an available displaced-surface review in Blender.
- Let the user record review decisions and notes for candidates. Quality review must include visible detail, relief placement, inversion, color-induced false bumps, grain, halos, seams, edge artifacts, smoothing, and displaced geometry under neutral and grazing illumination.
- An optional local adviser may recommend photo preparation, dataset suitability, or map-quality decisions. Suggestions must remain editable and require the user's choice before changing the material or review decision. Adviser scores must not establish physical accuracy or automatically select a released model.

## Saved models, packages, and transfers

- Provide a library of saved checkpoints with readable identities, target, training size, refinement scope, base identity, and training progress. Support locating existing checkpoints, selecting a model for generation, and opening the trainer with that model.
- Export a separate LoRA adapter in normal mode. Developer mode must also offer a complete fused material checkpoint that can run without separately installed base weights. Model packages must include the configuration, identity/integrity information, and required license notices needed to use and identify them.
- Allow mixing compatible adapters with explicit editable weights. Incompatible bases, targets, scopes, or trained-module layouts must be rejected before producing a package.
- Download base models and saved remote packages only on request. Show progress and allow cancellation. Verify that a download is complete and matches its declared identity before making it usable. App-owned downloaded base weights must be removable and recoverable from their recorded origin.
- Support Hugging Face account sign-in, account refresh, repository choice, public/private visibility, package upload, and a catalog of uploaded models with download actions. Default uploads to private. Any automatic upload after training must be explicitly enabled and show its destination and visibility before the run.
- Upload only the chosen model package. Do not upload source photographs, dataset maps, private tokens, or unrelated workspace files. Core dataset, inspection, generation, and training workflows must remain usable locally without an upload account.
- Settings must provide the working-folder location, local model location, preparation-worker count, developer mode, and optional account credentials. Credentials must not be exposed in logs or ordinary exported artifacts. No separate interpreter/runtime installation should be required for ordinary use.

## Required acceptance behavior

- Deliver a usable application, not untested controls. Every visible interactive control must be exercised in the actual application before it is presented as working. Verify its visible result, enabled/disabled states, keyboard behavior where relevant, and persistence where applicable.
- Demonstrate folder-arrow selection and highlighting at both subject and nested-folder levels; adjacent image loading; map icons and actions below the image; map switching; review changes; import; training preparation; cancellation; checkpoint save; comparison; and export with real source files.
- Demonstrate that displacement preparation contains no normal or roughness samples, that diffuse-only preparation contains no unrelated maps, and that repeated preparation of unchanged data does not duplicate work or files.
- Demonstrate preparation-worker editing and persistence, a core-count default on the actual machine, and bounded preparation memory on a large dataset. A machine crash, memory exhaustion, silent resolution reduction, damaged source file, inactive selection control, or incorrect sample role is a release-blocking defect.
- Ordinary supported workflows must complete without crashes, assertions, stale UI, or unexplained errors. Exceptional conditions such as missing sources, insufficient resources, and interrupted transfers must produce clear, recoverable outcomes while preserving user data.
