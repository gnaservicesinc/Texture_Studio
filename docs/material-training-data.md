# Native material training data

The source folder is the dataset. Current files and their material/map tags are authoritative; ordinary dataset opening scans that folder even if an old `dataset.json` is present. A saved registration cannot introduce an outside material, retain a removed material or hide a current image. Unassigned images, smaller maps and inspection errors remain visible in Originals. Training eligibility is separate from membership.

## Dataset actions

Use **Choose Source Folder…** to select existing originals, or **New Source Folder…** to create an empty folder. Keep each material's images and maps in its own subfolder. **Add Images or Folders…** copies outside sources into the selected folder without changing their pixels. Manual map assignment uses files in that folder; material category tags and map roles determine their training use.

Loom Room derives categories from the current tagged diffuse images. A training category includes matching material families and their available selected target maps. Browsing an original does not change that scope. Returning to the Trainer restores the selected category and map, and preparing a run refreshes their file membership.

Ordinary source-folder setup uses the maps and resolutions actually present. It does not require a fixed set of downloaded tiers. Missing targets remain explicit: a displacement map is never invented from bump or promoted from 8-bit data. Such materials remain visible and can train their available targets. Ambiguous map roles need explicit selection. Height training requires 16-bit scalar data or identical RGB channels; normal and diffuse inputs require suitable RGB channels.

Approve or exclude a crop and record a note. Reviews are bound to original map checksums and the selected grid; exclusion affects training without removing folder membership. Moving a material folder out removes its membership while keeping its originals together. Saved models, reviews and temporary preparation data do not redefine which files belong to the source folder.

## Native grids and numerical values

Supported grids are 256, 512, 1024 and 2048 pixels. Select the grid explicitly; inspection and navigation do not silently replace it. Exact-size originals remain direct references. Larger paired sources provide integer-aligned crops; rectangular maps qualify when both axes fit. Smaller sources remain in the inspector and are omitted from that training run. A folder containing 1K, 2K, 4K and 8K versions retains every set, while a 2K run uses fitting pairs. Random crop positions refresh at the selected run or epoch interval independently of optional flips and quarter-turn rotations.

Coordinates match across every map and diffuse color. Integer codes are sliced exactly: no resizing, padding, gamma conversion, range stretching or denoising. DirectX normal conversion complements the green integer code and preserves the remaining channels. Full native maps remain unchanged on disk, with any required convention conversion occurring at the model boundary.

Height, roughness and normal training tensors are planar Float32 values obtained from native codes divided by 65535 or 255 as appropriate. Targets receive no gamma or per-crop normalization. RGB compact models use recorded linear sRGB diffuse input; other families follow their recorded input contract. Source alpha and supported transparency keys remain in exact crops and do not determine crop position or alter numeric map values.

## Storage and concurrent changes

Original compressed images stay in the source folder. Category manifests and sample records are disposable projections, refreshed from current files and tags. Exact crops and run-owned augmented images live in app-managed cache or workspace storage outside the source folder. Each run prepares only its diffuse variants and selected displacement, roughness or normal target. Excluded crops and sets without the requested target produce no training images.

Derived records bind the current file inventory, source checksums, precision, dimensions, review state and crop geometry. They are evidence for a run, not an editable membership database. Color profile bytes remain in the originals; compact metadata records encoding and provenance without embedding those image bytes.

Preparation processes independent materials concurrently. Settings expose Preparation workers, with a default equal to the available CPU cores. The effective worker count also respects source count and a conservative working-memory budget. Each worker memory-maps one selected original at a time, inflates it scanline by scanline and retains only the requested crops; no full-resolution filtered or decoded image buffer is allocated. The estimate includes mapped compressed bytes, all retained crop pixels, encoder buffers, scanlines and overhead. A map that cannot fit the budget is rejected before starting workers. The activity text reports completed materials and worker count. Cancellation stops and joins every writer before releasing dataset locks or removing incomplete staging files. Lossless PNG encoding preserves integer codes; pixel-exact regression fixtures cover every PNG row filter, Adam7 pass, native precision and supported channel count. Preparation and cache validation use unchanged file size and modification/change timestamps to reuse recorded identity. The header is still checked on every validation; changed or missing file state requires a fresh SHA-256 check, and a successful check refreshes its stored state. This applies to both exact original references and generated crops.

Source-folder inspection reads current files without acquiring a dataset edit lock. Mutable derived metadata uses advisory locks and a durable journal; training holds shared locks on its app-owned preparation records. These locks protect run evidence and generated files, while source-folder membership continues to come from disk. Interrupted metadata writes recover only when the prior revision and replacement checksum still match.

Cleanup checks the exact generated files against their manifests before removing them. It preserves newer source reviews and refuses unexpected files or symbolic links. Review reconstructs the selected grid from checksum-bound originals. Export Original copies and verifies the original compressed bytes independently of display contrast.

## Colors and learning checks

Each actual resolution and source family remains distinct. Registered diffuse colors share the same geometry and surface targets. Provider names such as diffuse, color, albedo, numbered colors, OpenGL/DirectX normals and ambientCG resolution names are recognized. Specular, AO and other supplemental maps remain visible with the originals, even when they are not a training target.

Automatic diagnostics choose different subject views while honoring enabled folders and the percentage/crop limit. The percentage is a strict cap rounded down. Larger original views are preferred, and common subject/family coordinates prevent selecting an existing training view as a diagnostic. A different view may overlap training pixels; this is a known-subject learning check, not a claim of unseen-material accuracy. All target plans share region membership and differ only in available target counts.

Inspect original maps, reference/base/refined outputs and displaced surfaces. Preview labels and large exports cannot establish source precision or physical accuracy. Preserve source files and their provenance. A material enters or leaves the dataset when its files enter or leave the source folder; derived inventories must follow that change.
