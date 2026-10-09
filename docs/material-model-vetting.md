# Material model quality and export contract

The current development backend uses the complete revision-pinned PBRnxt material network, adapted to map input and output pixels at the same native grid. The original pretrained generator and final output branches are loaded strictly. The adaptation is experimental; it must earn acceptance for this surface-material task through actual maps and displaced-surface review.

## Training and model identity

The model receives a prepared diffuse map and one registered target: height, roughness or normal. Every pair matches the chosen training grid. There is no hidden smaller crop, per-image range stretching, numeric gamma correction, or fabricated enlargement of undersized targets. Training loss includes the complete target grid.

LoRA refinement targets the selected map's final branch, optionally including its decoder. The base identity, target, scope, rank, alpha, module layout, training size and step are stored in checkpoint metadata. Full and adapter checkpoints use `.safetensors`. A fused full checkpoint retains the separate LoRA so the changes can be examined and reproduced against the recorded base.

Weighted adapter mixing requires the exact same base identity and compatible target modules. Fusion adds the weighted adapter deltas to the base. Tests verify that the fused network agrees with the base-plus-adapter network within floating-point arithmetic tolerance; a saved file format cannot make two separately evaluated operation orders bitwise identical.

Developer mode exposes controls for this refinement workflow and defaults to a full fused model plus LoRA. User mode exports the LoRA. A future V1 material catalog is intended to use validated refined full models as the starting points for further task-specific refinement.

## Acceptance

Inspect maps at the actual model input resolution. Compare the pretrained base, source reference and result on the same diffuse pixels. Examine relief placement, inversion, grain, false bumps from color, halos, edge frames, seams and excessive smoothing. Inspect displaced geometry under neutral and grazing light. A lower fitting loss or larger output file does not establish improved material quality.

Automatic checks hold out source families at the selected grid, keeping all resolutions and colors together. Regional checks of disjoint corners of a known material are labeled separately. Fresh materials and photographs prepared through the application's diffuse process are still required to assess use outside the training set. A one-region fitting run has no independent validation set.

The local decision model can rank detail, visual appeal and visible artifacts and preselect a recommendation. Its output is advisory and retains the user's final review choice. Neither a decision score nor a numerical metric automatically promotes a model into the shipped catalog.

## Precision, memory and recovery

Height uses original UInt16 numeric codes transferred to Float32 by division by 65535. Roughness and normal maps use their actual integer precision. Raw predictions export lossless unclipped Float32 EXR; display PNGs are separate derivatives. Source images remain unchanged.

Memory preflight determines which training grids are offered for the selected scope and budget. Generation may tile larger prepared diffuse maps independently of training size. Hardware estimates and a successful run describe runtime feasibility, not model quality.

Hub exports retain model metadata, licenses, source files and base identity and exclude source photographs and optimizer state. Successfully uploaded models remain in the app's download catalog by repository revision. Local full checkpoints can be used as a different base without silently changing architecture or tensor layout.
