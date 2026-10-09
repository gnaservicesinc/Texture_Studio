# Native material training data

Material Dataset stores checksum-bound references to original diffuse, height, roughness and normal PNGs. Registration preserves their dimensions, integer precision, color metadata and recorded provenance. Swift inspects PNG headers and binds original files with one SHA-256 checksum. Its integer decoder reverses compression, filters and Adam7 without a display color pipeline. Independent precision tests compare exported crops to the selected original integer pixels.

## Dataset actions

Create a named dataset with New Dataset, or open its folder. Dataset Info edits its name, description, native grid and validation settings. Add Materials assigns a diffuse map and at least one surface map. Import Folder recognizes paired provider maps or explicit diffuse/height/roughness/normal names and shows a preview before registration. Sources are referenced without copying or rescaling them.

Every map and color variant in a source set must have the same actual dimensions. Registration rejects ambiguous role assignments. Normal and diffuse inputs require RGB or RGBA channels. Height training requires 16-bit scalar data or identical RGB channels; a missing displacement map is never invented from bump or promoted from 8-bit data. Such sets remain useful for their available targets.

Approve or exclude a crop and record a note. Reviews are bound to the original map checksums and selected grid. Removing a material removes membership while keeping its files. Deleting a dataset uses macOS Trash after checking ownership, current metadata revision and every file in the proposed scope. A folder containing originals, unrelated files or unreadable entries is not authorized for whole-folder deletion.

## Native grids and numerical values

Supported grids are 256, 512, 1024 and 2048 pixels. The selected size applies to all diffuse variants, targets and diagnostic views. Exact-size originals remain direct references. Larger sources produce one centered native crop. Sources at least 8K on both axes produce top-left, top-right and bottom-right crops. Rectangular maps qualify when both axes fit the grid. Smaller sources remain in the inspector and are omitted from that training run.

Coordinates match across every map and diffuse color. Integer codes are sliced exactly: no resizing, padding, gamma conversion, range stretching or denoising. DirectX normal conversion complements the green integer code and preserves the remaining channels. Full native maps remain unchanged on disk, with any required convention conversion occurring at the model boundary.

Height, roughness and normal training tensors are planar Float32 values obtained from native codes divided by 65535 or 255 as appropriate. Targets receive no gamma or per-crop normalization. Diffuse input declares sRGB or linear encoding; linear input is explicitly converted for the model's sRGB input contract. Alpha must be opaque for training. Transparency-table PNGs require explicit channel conversion before registration rather than losing transparency silently.

## Storage and concurrent changes

The source folder keeps its original compressed files. Only actual crops write generated PNGs beneath the source dataset's owned `.training-data/` directory. Each run prepares diffuse color variants and its selected height, roughness or normal target. A displacement run does not open or export normal or roughness maps. Excluded crops and source sets without the requested target produce no training files. Import registers available original map references for inspection and future targets, without creating any image copies.

Compact JSON records file references, the dataset revision, source precision, actual dimensions, review state and crop geometry needed to distinguish validation views. Color profile bytes remain inside the PNG rather than appearing as base64 text in manifests. Gamma and declared color encoding remain explicit. Redundant profile payloads, extra MD5 fields, repeated source-file inventories and normalized coordinate copies from earlier manifests are removed when the dataset opens. SHA-512 is not used. An official provider audit may check its published MD5 once without adding a second per-map content hash.

Preparation processes independent materials concurrently. Settings expose Preparation workers, with a default equal to the available CPU cores. The effective worker count also respects source count and a conservative working-memory budget. Each worker memory-maps one selected original at a time, inflates it scanline by scanline and retains only the requested crops; no full-resolution filtered or decoded image buffer is allocated. The estimate includes mapped compressed bytes, all retained crop pixels, encoder buffers, scanlines and overhead. A map that cannot fit the budget is rejected before starting workers. The activity text reports completed materials and worker count. Cancellation stops and joins every writer before releasing dataset locks or removing incomplete staging files. Lossless PNG encoding preserves integer codes; pixel-exact regression fixtures cover every PNG row filter, Adam7 pass, native precision and supported channel count. Preparation and cache validation use unchanged file size and modification/change timestamps to reuse recorded identity. The header is still checked on every validation; changed or missing file state requires a fresh SHA-256 check, and a successful check refreshes its stored state. This applies to both exact original references and generated crops.

Native operations use advisory dataset locks and a durable metadata journal. Replacement metadata is staged as ordinary JSON files; the recovery journal stores paths and revision checksums, never base64 copies of replacement bytes. Legacy journals remain recoverable. Interrupted metadata writes recover only when both the prior revision and planned replacement checksum still match. A training operation holds shared locks on its prepared and original datasets; editing or deletion from another window becomes available after that training operation finishes.

Cleanup checks the exact generated files against their manifests before removing them. It preserves newer source reviews and refuses unexpected files or symbolic links. Review reconstructs the selected grid from checksum-bound originals. Export Original copies and verifies the original compressed bytes independently of display contrast.

## Colors and learning checks

Each actual resolution and source family remains distinct. Registered diffuse colors share the same geometry and surface targets. Provider names such as diffuse, color, albedo, numbered colors, OpenGL/DirectX normals and ambientCG resolution names are recognized. Specular, AO and unsupported maps are omitted with import information.

Automatic diagnostics choose different subject views while honoring enabled folders and the percentage/crop limit. The percentage is a strict cap rounded down. Larger original views are preferred, and common subject/family coordinates prevent selecting an existing training view as a diagnostic. A different view may overlap training pixels; this is a known-subject learning check, not a claim of unseen-material accuracy. All target plans share region membership and differ only in available target counts.

Inspect original maps, reference/base/refined outputs and displaced surfaces. Preview labels and large exports cannot establish source precision or physical accuracy. Preserve source files and their provenance. Missing-source membership can be removed only after a recorded path is genuinely absent and no intact checksum-matching original can be recovered.
