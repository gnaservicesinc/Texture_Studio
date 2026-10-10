# Material model quality and export contract

The current development backend uses the complete revision-pinned PBRnxt material network, adapted to map input and output pixels at the same native grid. The original pretrained generator and final output branches are loaded strictly. The adaptation is experimental; it must earn acceptance for this surface-material task through actual maps and displaced-surface review.

## Training and model identity

The model receives a prepared diffuse map and one registered target: height, roughness or normal. Every pair matches the chosen training grid. There is no hidden smaller crop, per-image range stretching, numeric gamma correction, or fabricated enlargement of undersized targets. Training loss includes the complete target grid.

LoRA refinement targets the selected map's final branch, optionally including its decoder. The base identity, target, scope, rank, alpha, module layout, training size and step are stored in checkpoint metadata. Full and adapter checkpoints use `.safetensors`. A fused full checkpoint retains the separate LoRA so the changes can be examined and reproduced against the recorded base.

Weighted adapter mixing requires the exact same base identity and compatible target modules. Fusion adds the weighted adapter deltas to the base. Tests verify that the fused network agrees with the base-plus-adapter network within floating-point arithmetic tolerance; a saved file format cannot make two separately evaluated operation orders bitwise identical.

Developer mode exposes controls for this refinement workflow and defaults to a full fused model plus LoRA. User mode exports the LoRA. A future V1 material catalog is intended to use validated refined full models as the starting points for further task-specific refinement.

## Acceptance

For the current development experiment, acceptance means a visually realistic rendered surface. A generated map may differ substantially from its source reference and still be acceptable. Source-agreement losses remain training diagnostics; the user's visual review decides whether the result succeeds. Rank, alpha and training size remain experimental settings rather than a prescribed final configuration.

Inspect maps at the actual model input resolution. Compare the pretrained base, source reference and result on the same diffuse pixels. Examine relief placement, inversion, grain, false bumps from color, halos, edge frames, seams and excessive smoothing. Inspect displaced geometry under neutral and grazing light. A lower fitting loss or larger output file does not establish improved material quality.

Automatic checks hold out source families at the selected grid, keeping all resolutions and colors together. Regional checks of disjoint corners of a known material are labeled separately. Fresh materials and photographs prepared through the application's diffuse process are still required to assess use outside the training set. A one-region fitting run has no independent validation set.

The local decision model can rank detail, visual appeal and visible artifacts and preselect a recommendation. Its output is advisory and retains the user's final review choice. Neither a decision score nor a numerical metric automatically promotes a model into the shipped catalog.

## Precision, memory and recovery

Height uses original UInt16 numeric codes transferred to Float32 by division by 65535. Roughness and normal maps use their actual integer precision. Raw predictions export lossless unclipped Float32 EXR; display PNGs are separate derivatives. Source images remain unchanged.

The selected native grid is never reduced to fit a hardware estimate. Training and inference evaluate complete grids with Apple MPSGraph; actual working memory and throughput depend on the installed model and chosen map size. A successful run establishes runtime feasibility, not model quality.

Hub exports retain numeric weights, metadata, license notices and base identity and exclude executable sources, source photographs and optimizer state. Successfully uploaded models remain in the app's download catalog by exact repository revision. Local full checkpoints can be used as a different base without silently changing architecture or tensor layout.
