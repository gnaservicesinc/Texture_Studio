# Native Texture Studio workflow

Texture Studio prepares a surface photograph as a diffuse map and uses a refined material model or a registered source-height map to create editable material maps. Original files remain unchanged. JSON recipes and Blender exports provide file-based interoperability.

## Geometry and photographic preparation

Perspective controls use a camera-centered pinhole projection of a plane. Camera metadata can supply an approximate focal length; a user override is available. The square crop stays inside the projected footprint. Nonplanar surfaces remain a planar approximation.

ImageIO supplies the primary photo, camera metadata, gain maps and any registered spatial companions. The primary photograph anchors supporting-view registration. Exposure and color matching and agreement checks determine whether companion samples contribute; occlusions, parallax and poor matches keep the primary evidence.

Photographic processing uses extended linear sRGB and a Metal Float32 context. HDR color is decoded as a complete frame before geometry or crop operations. Broad illumination correction and a smooth photographic highlight rolloff prepare the diffuse map. The high-frequency photo detail is retained: the photo path does not apply a spatial denoiser. Companion evidence can reduce capture noise without inventing texture.

The finished diffuse pixels are frozen and shared by preview, export and material inference. Checkpoint testing invokes this same preparation. The model backend accepts a prepared diffuse PNG and rejects untreated photographic input. Illumination correction cannot guarantee reconstruction of a completely obscured shadow or saturated highlight; inspect the diffuse result before judging the model.

## Processing and precision

Numeric maps use a separate color-unmanaged Float32 context. Height, roughness and normals receive no color-profile gamma or photographic tone mapping. A map explicitly identified as **surface height** retains its supplied amplitude through registration and explicit relief controls. Normals derived from height use mathematical differences, material width and displacement amplitude and follow OpenGL +Y.

Embedded portrait depth does not supply material displacement. With no material model or attached height, displacement is flat. Photographic brightness relief remains an explicit artistic control rather than a default source of geometry.

Source bit depth, model arithmetic and export precision are recorded separately. Diffuse export is 16-bit sRGB PNG, using the same encoding as the model input. Numeric maps use HALF or FLOAT EXR according to the export choice; Float32 output uses the explicit lossless FLOAT writer. Changing container precision cannot add source or model detail.

Blender uses **Non-Color** for scalar height, roughness and encoded RGB normals. The generated setup script keeps geometric displacement and normal contribution explicit so equivalent height relief is not accidentally applied twice. Original exports remain separate from display contrast, previews and temporary training-grid views.

## Training and generation

The current experimental material network is a complete pinned PBRnxt adaptation with native input/output grids. It supports LoRA refinement for height, roughness and normals. Training consumes complete diffuse and target crop grids at exactly the selected dimensions. Registered diffuse colors can alternate while target geometry stays fixed. Automatic memory budgets follow resolution and retain system headroom; developer mode exposes a manual override. The UI offers only sizes allowed by the backend budget and actual source dimensions. Generation/export sizes are independently selectable.

Unchanged exact-size files use original source paths. Larger 4K sources use one center native crop; sources at least 8K use three disjoint corners. Crop coordinates are recorded and matched across all roles and diffuse colors. Source sets smaller than the selected grid are excluded. Only the final temporary training files are written. After a run or size switch, those owned files are removed, while small reviews remain and the original dataset returns to view. Inspection reconstructs the exact training grid from originals in memory when necessary.

Developer mode in Settings exposes refinement scope, adapter controls, base overrides and upload controls. It exports a fused full `.safetensors` checkpoint and keeps the separate LoRA. Normal mode exports the LoRA. Weighted adapter mixing requires the same recorded base, target and compatible module layout. Saved Hub models remain downloadable by their recorded repository revision.

## Local decision adviser

The optional local Clef adviser receives bounded display copies of photos or diffuse/output map pairs. It uses typed choices and scores to suggest preparation settings and review recommendations for detail, visual appeal and artifacts. Outputs are validated as decisions, not executable instructions. The user controls which recommendations are applied or saved.

The adviser runs through local Ollama and releases its model memory after requests. Its large model download is explicitly managed in the app. Numerical original maps are not replaced by the adviser's display images, and model scores do not establish geometric or physical accuracy.

## Quality acceptance

Compare source references, base and trained outputs on the same diffuse pixels at the real model grid. Inspect grain, bump placement, inversion, halos, seams, false relief from color, edge frames and smoothing. Inspect displaced geometry under neutral and grazing light. Automatic checks hold out complete source families across resolutions and colors. Disjoint regional checks are permitted for a single included resolution set and are labeled accordingly; a one-region fit has no independent validation set.

The planned V1 catalog uses validated refined full models as the bases for later task-specific refinement. That quality remains a separate acceptance decision. See [native material tools](native-material-tools.md), [data contract](material-training-data.md), and [model acceptance](material-model-vetting.md).
