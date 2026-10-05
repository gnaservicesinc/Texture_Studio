# Extraction, Photo Studio and Raw Studio

## Extract selected products

Import or drag a HEIC into Extractor and inspect its inventory. Select only the products you need. Raw depth, Apple native disparity/depth, HDR gain maps, portrait and semantic mattes, alpha planes, stereo views and display images depend on what that file actually contains. A missing auxiliary item cannot be reconstructed by extraction.

Raw extracted arrays stay in their stored coordinate system; recorded orientation does not imply that pixels have been rotated. Keep calibration with arrays when consuming them in another program. Optional NPY companions preserve exact sample representation. JSON manifests are optional and record inspection, calibration and provenance; important derived-product metadata is also stored in supported exchange files.

Enable RAFT or classical stereo only when estimating new geometry is your goal. RAFT requires the local source/runtime and a compatible checkpoint. Invalid explicitly selected model paths fail; model selection is not an implicit download. Reverse-consistency checks are additional computation. Classical estimates and their support masks use different evidence requirements; unsupported smooth surfaces are not automatically zero-height surfaces.

Extractor reads the selected checkpoint's metadata to choose its decoder, including renamed checkpoints. A display-depth student automatically supplies display-grid depth, displacement and preview products when those outputs are requested. Raw student depth retains its checkpoint units and float32 values; displacement and previews remain separate derivatives. Native disparity, signed-flow and stereo support diagnostics are offered for compatible RAFT checkpoints.

**Color Matching** is optional inference preprocessing. The selected Hero view remains the reference, and the other view's RGB distribution is matched. It does not edit original extracted arrays or provide a camera-profile calibration. **Tolerate camera detail differences** affects the classical matcher's inference copies. These controls are unnecessary when exporting only raw auxiliary planes.

## Photo Studio

Select a file, then an operation and product. **Original** exports the chosen plane. **Depth upscale** creates a separately labelled RGB-guided interpolation on a supported registered grid. **Learned depth** and **RAFT depth** create estimates. Inspect original metadata to establish which units and camera reference you selected.

For a people cutout, select the portrait and useful semantic layers. The composite uses their maximum coverage, which avoids artificially increasing opacity at overlaps. Check hair, skin, glasses and fine edges in the preview; deselect incorrect layers. Sky is not a people matte. **Isolate** exports a selected region. Color values stay separate from the new alpha computation. **Export & copy** copies exact PNG bytes where appropriate and file URLs for other formats.

HEIC sharing/repair is supported only when the available writer can preserve every required retained image, auxiliary and calibration item. A refused rewrite leaves the original intact. Privacy cleanup addresses supported recorded metadata; recognizable people and places remain visible in the picture. When a container cannot be repaired exactly, export separate lossless arrays for your workflow.

## Raw Studio

Use **Original** to preserve the source RAW file. Available sensor/LinearRaw exports retain the stored raster, source dtype and margins together with camera calibration. Some Apple ProRAW files contain demosaiced LinearRaw samples rather than a color-filter mosaic; do not assume every RAW file has the same sensor representation.

**Developed RGB** is a separate derived product. The native extended-linear RGB result can contain negative values and values above one. These are retained numerically. Teacher inference uses a separate model input copy. **Person mask** uses Apple Vision on a viewing copy of the corresponding developed grid. **Learned depth** generates a separate estimate.

The optional region is expressed in the selected product's pixel coordinates. Sensor coordinates, developed RGB coordinates and a depth model's native grid are not interchangeable. Preserve full sensor calibration before cropping or converting data externally. A preview is a convenience display, not the scientific output.
