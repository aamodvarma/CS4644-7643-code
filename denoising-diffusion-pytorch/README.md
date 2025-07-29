# Spatial Masked DDPM models
We work on LucidRains implementation of DDPM model and further add three additional features to this implementation,

1. Adaptive Noise Injection
2. Image generate with conditional masks
3. Spatially weighted loss function


## Preprocessing
To use the above three methods we first need to preprocess our images to include spatial masks. We support two kinds of masks, edge aware masks and sematic masks. Edge aware masks are automatically generated on the fly using filters while semantic masks needs to be generated before hand. `bisnet_mask.py` has the relevant code in order to generate these masks.

# Design Variants
The following are the relevant flags to change in order to use the above three features we included,
1. Adaptive Noise Injection: Make sure to have `adaptive=True` in the diffusion model and have `adaptive_mask_type` equal to either `edge_aware` or `semantic`.
2. Conditional generation using masks: Update `cond_channels` in the unet model to reflect the dimentions of the mask. For edge aware masks this should be set as `1` while for semantic masks this should be `19` (19 channel mask, relevant information about each channel can be seen in the `bisnet_maks.py` file). Also have `conditional_mask_type` in trainer as either `"edge_aware"` or `"semantic"` depending on the mask choice.
3. Spatially weighted loss: Have `masked_loss=True` in the diffusion model and have `masked_loss_type` as either `edge_aware` or `semantic` as required.
