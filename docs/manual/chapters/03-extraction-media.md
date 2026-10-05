# Extraction, Photo Studio and Raw Studio

## Extract selected products

Import or drag a HEIC into Extractor and inspect its inventory. Select only the products you need. Raw depth, Apple native disparity/depth, HDR gain maps, portrait and semantic mattes, alpha planes, stereo views and display images depend on what that file actually contains. A missing auxiliary item cannot be reconstructed by extraction.

Raw extracted arrays stay in their stored coordinate system; recorded orientation does not imply that pixels have been rotated. Keep calibration with arrays when consuming them in another program. Optional NPY companions preserve exact sample representation. JSON manifests are optional and record inspection, calibration and provenance; important derived-product metadata is also stored in supported exchange files.

The built-in depth sources are **DepthPro**, **DA3** and **DA2**: one source per model. DepthPro supports **Export checked**. DA3 and DA2 are individual exports: select one row and click **Export this map**. A cancel-first warning explains their long runtime before inference starts. DA3 took about 20 minutes for one 5712 × 4284 photo in testing; DA2 also takes minutes at native resolution. Each model uses the full display photo and saves one full-resolution **32-bit float EXR**, such as 5712 × 4284. The generated values receive no normalization, gamma correction, tone mapping, quantization or sharpening. There are no native-grid, preview or normalized duplicate AI sources.

**Default depth source** selects model settings. DepthPro is the preset's checked model; DA3 and DA2 remain individual exports. Advanced settings provide local checkpoint/source paths, device and an optional processing-size override; full display processing is the default for DA3 and DA2. DepthPro uses its required fixed internal model grid. Models must already be installed. Optional exact NPY companions and JSON manifests remain available. DepthPro estimates meters; DA2 retains relative inverse depth and DA3 retains relative depth.

RAFT requires the local source/runtime and a compatible checkpoint. Invalid explicitly selected model paths fail; model selection is not an implicit download. Classical generation has been removed from exports.

All RAFT choices stay visible when changing models: depth, 0–1 displacement, depth preview, display-grid depth and preview, native pixel disparity, signed flow, support and supported depth. Extractor reads checkpoint metadata to use the selected decoder, including renamed checkpoints. Display depth models supply their learned map for primary depth/displacement/preview exports, preserving checkpoint units and raw float32 values. Native disparity, signed flow and stereo support diagnostics use that same checkpoint's embedded RAFT component. Native correspondence checkpoints produce left-grid depth, with explicit display-grid products when requested. Each row describes the grid and units; changing a model does not replace it with another model.

RAFT numerical EXRs use one scalar channel, and RAFT previews use grayscale PNG without a redundant alpha channel. The support mask remains a separate selectable product. Displacement and previews are explicit viewing/effect derivatives; raw inferred depth is preserved separately.

**Color Matching** is optional inference preprocessing. The selected Hero view remains the reference, and the other view's RGB distribution is matched. It does not edit original extracted arrays or provide a camera-profile calibration. This control is unnecessary when exporting only raw auxiliary planes.

## Photo Studio

Select a file, then an operation and product. **Original** exports the chosen plane. **Depth upscale** creates a separately labelled RGB-guided interpolation on a supported registered grid. **Learned depth** and **RAFT depth** create estimates. Inspect original metadata to establish which units and camera reference you selected.

For a people cutout, select the portrait and useful semantic layers. The composite uses their maximum coverage, which avoids artificially increasing opacity at overlaps. Check hair, skin, glasses and fine edges in the preview; deselect incorrect layers. Sky is not a people matte. **Isolate** exports a selected region. Color values stay separate from the new alpha computation. **Export & copy** copies exact PNG bytes where appropriate and file URLs for other formats.

HEIC sharing/repair is supported only when the available writer can preserve every required retained image, auxiliary and calibration item. A refused rewrite leaves the original intact. Privacy cleanup addresses supported recorded metadata; recognizable people and places remain visible in the picture. When a container cannot be repaired exactly, export separate lossless arrays for your workflow.

## Raw Studio

Use **Original** to preserve the source RAW file. Available sensor/LinearRaw exports retain the stored raster, source dtype and margins together with camera calibration. Some Apple ProRAW files contain demosaiced LinearRaw samples rather than a color-filter mosaic; do not assume every RAW file has the same sensor representation.

**Developed RGB** is a separate derived product. The native extended-linear RGB result can contain negative values and values above one. These are retained numerically. Teacher inference uses a separate model input copy. **Person mask** uses Apple Vision on a viewing copy of the corresponding developed grid. **Learned depth** generates a separate estimate.

The optional region is expressed in the selected product's pixel coordinates. Sensor coordinates, developed RGB coordinates and a depth model's native grid are not interchangeable. Preserve full sensor calibration before cropping or converting data externally. A preview is a convenience display, not the scientific output.
