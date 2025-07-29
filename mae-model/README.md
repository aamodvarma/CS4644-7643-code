# Structured Masking for Facial Image Inpainting

This repository investigates how structured occlusion masks affect inpainting performance using Masked Autoencoders (MAE), and image generation using Diffusion Models.

---

## Base Directory (`./`)

Shared filtering and data processing code used across both MAE and DDPM approaches.

- `00_ffhq_proc.ipynb`:  
  Filters FFHQ images using MediaPipe and dlib landmarks  
  Flags and removes occluded or side-profile faces  
  Outputs: `clean_128/`, `image_occlusion_labels.csv`, `image_split.csv`

*Note: FFHQ images are not included and can be downloaded from the [official dataset repo](https://github.com/NVlabs/ffhq-dataset).*

---

## MAE Codebase (`mae-1/`)

Implements finetuning and evaluation for MAE under different masking strategies.

### Preprocessing

- `01_structuredmask.ipynb`:  
  Generates structured masks:  
  - Edge-based (Sobel filters)  
  - Region-based (eyes, nose+mouth, all) via MediaPipe  
  Masks are dilated to cover 25%, 50%, or 75% of the image and saved in a MAE-compatible folder structure. Train-inference splits are also generated in this file.

### Finetuning

- `main_finetune_structured.py`, `engine_finetune_structured.py`:  
  Finetune MAE on structured masks  
- `main_finetune_rand.py`, `engine_finetune_rand.py`:  
  Finetune MAE with random masking (25/50/75%)  
- `datasets/masked_faces_dataset.py`:  
  Custom PyTorch `Dataset` for training with image-mask pairs

### Inference and Evaluation

- `infer_faces_struct.py`, `eval_recons_struct.py`:  
  Inference and evaluation for structured-mask models  
- `infer_faces_randXXX.py`, `eval_recons_randXXX.py`:  
  Inference/evaluation for random-mask variants  
- `eval_recons_base.py`:  
  Evaluation of the pretrained baseline MAE

### Utilities

- `models_mae.py`: MAE encoder-decoder model  
- `models_vit.py`: ViT backbone  
- `util/`, `engine_pretrain.py`, `engine_finetune.py`: Utilities for training/evaluation  
- `run_*.sh`: Slurm scripts for job submission on PACE cluster

---

### Output Directories

Each output folder corresponds to a model variant (e.g., `outputs_rand025/`, `outputs_structured_finetune/`) and contains subfolders for each mask type used during inference.

Inside each mask-specific folder (e.g., `masks_edge_50_thresh50/`):

- `masked/`: Occluded input images  
- `nonmask+recon/`: Original + reconstructed outputs  
- `avg_patch_mask_coverage.txt`: Average patch-level occlusion coverage  
- `summary_metrics*.txt`: Evaluation metrics (FID, PSNR, SSIM, LPIPS, ArcFace cosine similarity)

*Note: Image files are excluded from this repo.*

---

## DDPM Folder

(Placeholder)

- Conditional DDPM models trained on structured mask variants  
- Experiments with noise-weighted loss and edge/semantic-conditioned U-Nets  
- Adaptive noising and blur strategies for corruption

---

## References

- Facebook Research. *Masked Autoencoders for Visual Pretraining*.  
  https://github.com/facebookresearch/mae, 2021.
- NVLabs. *FFHQ Dataset*.  
  https://github.com/NVlabs/ffhq-dataset
