# Photo Studio and Raw Studio

Photo Studio works with the full HEIC color image, Apple depth/disparity,
portrait and semantic people mattes, HDR gain maps and other auxiliary images.
Raw Studio preserves the original RAW file, exposes stored RAW rasters and
calibration, and creates separately labelled color/depth/mask derivatives.
Both use `media_studio.py` / `ipde.media_cli` as their JSON backend.

## Precision and grids

Original NPY, TIFF, PNG (unsigned 8/16-bit) and OpenEXR (float16/float32) exports
are decoded and compared bit-for-bit before installation. No original array
is normalized, gamma corrected, tone mapped, demosaiced or resized. NPY retains
arbitrary numeric dtypes and NaN payload bits. Choosing PNG for floating depth
fails rather than silently converting it. All multi-product exports remove
completed products if a later output fails; existing outputs are never replaced.
Original HEIC arrays preserve decoded sample codes and their encoded transfer
characteristics; a decoded code value is not automatically linear-light RGB.

An upscaled embedded portrait map is an **RGB-guided interpolation**, labelled
as derived. It adds no measured precision. It retains invalid regions as NaN,
keeps the embedded representation/accuracy, and refuses a different camera or
unknown aspect registration. Its original lower-resolution array stays intact.
DepthPro, Depth Anything and RAFT produce estimated, derived depth. The backend
saves native model predictions separately from full source-grid results.
Relative teacher outputs require calibration before they can represent metric
repair data. A full-resolution grid is not a full-resolution measurement.

People matte composition takes the maximum coverage of the selected portrait,
skin, hair, teeth, glasses or eyes layers. Overlaps do not artificially increase
opacity. Sky is not a people matte. The output uses straight alpha, retains each
RGB sample, preserves the source ICC profile in PNG/TIFF, and computes only the
new alpha channel. Preview the layers because
Apple mattes occasionally misidentify people. Layer removal is selection based;
original embedded planes remain available.

RAW sensor/LinearRaw exports retain the full raster, source dtype and margins.
LibRaw camera black/white bounds, white balance, CFA, orientation and color
calibration are exported as companion JSON. DNG TIFF/SubIFD tags are retained,
including rational pairs, opaque byte blocks and unapplied linearization tables.
Apple ProRAW may store demosaiced **LinearRaw codes**, rather than a CFA mosaic.
The decoder preserves those stored codes. For example, a 10-bit JPEG XL raster
can contain code 1024; the backend preserves it rather than clipping to 1023.
Its uint16 storage is not a claim of 16-bit measured precision. `imagecodecs`
provides JPEG XL decoding when LibRaw cannot open the DNG.

Native processed RAW RGB is an explicitly derived extended-linear-sRGB float32
image. It may contain negative values and values above one; its exported array
retains both. Model preprocessing clips only a separate RGB copy. Vision person
segmentation consumes a display RGB copy of this exact rendered grid, so RAW
orientation/cropping cannot accidentally mirror its mask. Its native mask is
retained numerically then resampled explicitly to that reference grid. RAW crops
are in the selected product's pixel grid; sensor and processed RGB coordinates
must not be interchanged.

## HEIC rewriting and privacy

Every rewritten candidate is isolated until the backend reopens it through
pillow-heif and native ImageIO. It checks the full color/auxiliary inventory,
source precision, all retained decoded bits, native floating depth values,
calibration XMP, primary-image identity and spatial calibration/structure.
Replaced native depth/people matte/gain-map planes must match their registered
native grid and representation. A full display-grid depth may be explicitly
resampled to its own embedded grid. `--replacement-kind depth` means meters;
`disparity` means inverse meters; `native` retains the selected representation.
Invalid/nonpositive physical replacements are refused. Encoded depth maps,
opaque auxiliary layouts and main color/subimage replacement are refused when
the available writer cannot preserve all original samples.

ImageIO source-to-destination copy is attempted without recompression. Its
metadata flags can be ignored for HEIC: on the checked portrait fixture, GPS,
timestamps and device MakerApple data remained. The backend refuses that result.
A supported BMFF metadata-only scrub preserves compressed image/auxiliary
payloads, removes unnecessary identifying metadata, scrambles names/UUIDs, and
runs the same independent decoded/structural verification. Unsupported boxes,
unknown personal metadata, altered grids or dropped auxiliary images fail
closed. No approximate RGB-only HEIC fallback is exposed as success.

ImageIO auxiliary replacement requires its native source-image writer. Some
containers cannot retain every auxiliary or decoded pixel with that writer.
Photo Studio probes the current container and disables unavailable repair.
The inspected `IMG_6678.HEIC` retained all samples during extraction but native
repair dropped auxiliary images and changed retained color/depth samples.
The two inspected spatial fixtures also lost stereo metadata. Even supplying
the unchanged native depth array caused changes. Repair was refused and no
output installed; restoring only the missing auxiliaries would not fix this.
Independent lossless image/array exports remain available.

Privacy enhancement concerns recorded metadata. Visible people, objects and
locations in the picture remain identifiable. Calibration required for valid
portrait/spatial operation is retained; unknown functional structures cause a
refusal instead of an unverified privacy claim.

## Examples

```sh
python media_studio.py photo-inspect portrait.HEIC --preview-dir /tmp/photo-preview
python media_studio.py photo-export portrait.HEIC --output-dir output --operation original --asset 7 --format npy
python media_studio.py photo-export portrait.HEIC --output-dir output --operation depth-upscale --format exr
python media_studio.py photo-export portrait.HEIC --output-dir output --operation cutout --matte 3 --matte 4 --format png
python media_studio.py photo-export portrait.HEIC --output-dir output --operation isolate --asset 4 --format tiff
python media_studio.py photo-rewrite portrait.HEIC --output-dir output --privacy
python media_studio.py photo-rewrite portrait.HEIC --output-dir output --replacement 7=depth.npy --replacement-kind depth
python media_studio.py raw-inspect image.DNG --preview-dir /tmp/raw-preview
python media_studio.py raw-export image.DNG --output-dir output --operation original
python media_studio.py raw-export image.DNG --output-dir output --operation sensor --crop 200,100,512,384 --format npy
python media_studio.py raw-export image.DNG --output-dir output --operation learned-depth --model depthpro --crop 0,0,1024,1024 --format exr
python media_studio.py raw-export image.DNG --output-dir output --operation person-mask --format npy
```

Model weights and source directories use the existing local trainer lookup.
`--model-path`, `--model-source-dir`, `--device`, `--raft-model`, `--raft-root` and
`--raft-model-member` select explicit resources. No model is downloaded
implicitly. The macOS build bundles `media-bridge`; source checkouts can compile
it with Xcode tools. Exact RAW sensor support uses `rawpy`, `tifffile` and
`imagecodecs`; native ImageIO provides camera RGB and additional data, while
Apple Vision provides an optional accurate person mask.
