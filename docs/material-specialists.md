# Specialist material models

Open **Material Tools → Model Training → Specialist Models**. The guided workflow
creates category datasets from original paired sources, then lets you train and
review each displacement, normal or roughness model separately.

1. **Create / Refresh Specialists** uses the bundled user-trained Diffuse Texture
   Type Classifier IFPv2. Expand the settings to type or choose another Core ML
   export, original material dataset, or specialist workspace. Additional paired
   sources use the classifier source folder configured in Dataset Studio.
2. Choose a category and map. **Set Up Training** opens the existing Trainer with
   a native compact model, that category's dataset and a descriptive model name.
   Set resolution, updates, learning rate, optional flips and quarter-turn
   rotations there, then start training. Source pixels are cropped without scaling.
3. A successfully trained checkpoint is recorded with its category, target,
   original membership, model hash, resolution and completed updates. Finishing
   training does not approve the model. **Continue Saved Model** resumes it.
4. **Compare with References** opens the same paired dataset and saved checkpoint.
   Inspect relief, contrast, artifacts and consistency; return using **Specialist
   Workflow**. **Approve for Generation** makes that reviewed model available to
   automatic routing. Changes to source membership require an updated model.
5. **Next Specialist** advances to a category with paired targets that still needs
   a current approved model for the selected map. Selection and workspace settings
   persist across app launches.

The initial live workspace is
`~/Library/Application Support/Texture Studio/Material Workspace/Specialist Models`.
`specialists.json` records the classifier identity, class assignments with
probabilities, category datasets, classifier-only examples, target availability
and saved model histories. Each category has a regular editable Dataset Studio
dataset under `datasets/`; originals are referenced in place and never resized.

Refresh preserves user exclusions, removals, manually added paired materials,
custom specialists, model records and classifier-only images. Unrelated occupied
workspace/dataset folders are rejected without replacement. The main dataset
membership is unchanged by discovering extra source-folder pairs. New explicit
AmbientCG and authored role filenames are supported, including
`Paper003_1K-PNG_Color.png` and `Cardboard_01_height_1K.png`. Grayscale diffuse
files retain their original channel layout and precision; the model receives a
three-channel copy in memory.

**Add Category / Subtype** creates another specialist from a classifier category
and optional subtype/filename text, for example Wood plus `planks`. Descriptive
source names and provider subtype/tag evidence select those references. A parent
class prediction does not automatically select a narrower subtype; use the saved
subtype model directly, or expand and retrain the classifier to recognize it.

In Texture Studio's inspector, choose **Automatic specialist models**. The
prepared diffuse is classified once, and matching approved complete native
models generate the available surface maps. The default confidence threshold is
0.7, directly editable in the specialist workflow. Unapproved models, changed
checkpoint bytes, stale source membership and classifier changes are skipped.
Choose an explicit fallback for missing or uncertain height categories: flat
surface, selected material model, or local depth model. A missing local fallback
is reported and uses flat height. Roughness and normals without specialist models
use the existing material generation path.

Recipes pin the actual chosen specialist checkpoint identities and classifier
decision. Exports include per-map checkpoint provenance and
`specialist-classification.json` / `specialist-routing.json`. Numerical surface
maps continue to use linear data values and Float32 inference/export; diffuse
classifier preprocessing is separate from exact reconstruction training crops.

The bundled classifier has 15 labels, including Bark, Paper, Drywall and Leather.
Its source SHA-256 is
`e2936fcef0781fb4f85eb7e879cc2f7d865528415180d894afd74eddc1e048be`.
Final live preparation produced 375 native sets across 122 material families,
including the seven Paper003/Paper005/Cardboard_01 sets and three additional
Cardboard002 sets recognized by the expanded filename parser. It retained 24
classifier-only examples.
Cardboard002 was predicted as Leather at approximately 79% confidence; the
classifier dataset refresh adds it to Paper using its explicit material name.
The matching curated Paper label controls specialist dataset membership. Retrain
and select an updated classifier export to improve its image-based routing.
Fourteen categories have paired targets ready to train. Other Diffuse Textures
currently contains classifier examples only; add paired material maps before
training its specialist. No production specialist weights are claimed to have
been learned by this setup.
