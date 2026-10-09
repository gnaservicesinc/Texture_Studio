# Curated material training data

The material model learns from registered diffuse, height, roughness and normal maps. Diffuse is the model input. Numeric targets retain their declared source precision and encoding; height requires UInt16 linear codes. PNG headers, complete checksums, decoded arrays and lossless export round trips establish what the model receives.

## Native pixels at one grid

Each original resolution is a separate registered set. A selected training size applies to every diffuse variant, target and validation sample. For a 2048 grid, input, target and review are all 2048×2048; every pixel participates in the loss. All 1K files are ignored for that run, without upscaling.

Exact-size originals are referenced directly. Larger 2K/4K sources produce one centered native pixel crop at the selected grid. Sources at least 8K on both axes supply three disjoint corner crops selected from four corners. An eligible regional validation assignment uses two training corners and one check corner. Every role and color variant uses identical recorded coordinates; numeric pixels are sliced exactly, without resizing, padding, gamma changes or denoising. Rectangular sources qualify only when both axes fit the selected crop.

Height, roughness and normal channels use native integer code divided by that integer type's maximum when creating Float32 tensors. They receive no gamma, min/max normalization, tone mapping or denoising. Color input has a separately declared transfer. DirectX normals are converted by reversing the green component; exact-size sources stay on disk unchanged and the conversion occurs in memory.

## Storage

Original compressed files remain in the source folder. Complete maps already at the selected grid are referenced directly by absolute path and source checksum. No clone, hard link or duplicate image is created. Only intentional native pixel crops write final training images, under the app-owned `.training-data/` folder.

One active training view is retained. Switching size, finishing training or stopping a run removes the owned prepared files. Small per-size curation records survive cleanup. Review reconstructs the selected grid from checksum-bound originals instead of keeping extra full image copies. Original export remains byte-for-byte.

The local collection at `/opt/ipde/material-dataset` is refreshed from `/opt/ipde/material-dataset/sources`. Adding published 2K originals or color variations adds small manifests, without copying existing 4K/8K files. Downloaded whole-material framing sets must match the selected training resolution. Publisher checksums and original precision are verified. Model weights, dataset files and generated outputs are excluded from Git.

## Color variants and validation

Old `col`, `coll`, `color`, `albedo`, numbered color names and current diffuse names are recognized. Training chooses a registered color variant with the seeded training RNG each update, using the same displacement/roughness/normal pixels. Validation chooses deterministically; comparisons give base and refined models the same diffuse variant. Specular, AO, bump and other unsupported maps are ignored. Missing displacement is never invented from bump or promoted from 8-bit data; such sets remain usable for their available targets.

## Review and validation

Approve or exclude individual native crops and record specific problems. Check registration, grain, relief location, inversion, halos, seams and transparency. Display previews cannot establish numeric precision. Review both original-resolution maps and displaced surfaces under useful lighting.

Automatic validation holds out source families together across resolutions and color variants. Regional validation is allowed only for a family with one included resolution set and disjoint native crops; it measures held-out regions of a known material, rather than unseen material generalization. A single-region quick fit has no independent check set. The optional decision model can recommend a review outcome from diffuse and predicted maps; its recommendation does not replace the user's saved decision.

Keep downloaded source files and their provenance. A missing staged image is reproducible from its source; a missing original is removed from the list only when no intact matching source can be found. Header labels or high-resolution export sizes cannot add source detail.
