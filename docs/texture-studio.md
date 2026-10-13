# Texture Studio

Import a photograph of a surface. Straighten the image, choose a crop and adjust broad lighting balance in the inspector. Generate Material creates diffuse, displacement, roughness and OpenGL normal maps from that same prepared image.

Texture Loom is fixed in the application build. Its classifier selects a material dataset and the matching bundled submodels. The LoRA panel imports compatible Loom Workshop exports and turns their material overrides on or off. Replacing the base model requires rebuilding Texture Studio with a Loom Factory export.

Choose Attached Height / Depth Map to supply a registered map, or Flat Surface when no inferred relief is needed. The flat option creates neutral displacement and normals unless artistic brightness relief is enabled.

Open Full Quality inspects the generated map at native resolution. Export Material writes all four maps, linear EXR data, a 16-bit sRGB diffuse PNG and a Blender setup script into a new folder. Existing export folders are preserved. Source photographs are never rewritten.

Recipes save source paths, material settings, the built-in Texture Loom version and enabled LoRA identities. A recipe that requires another model build or missing LoRA is identified before generation.
