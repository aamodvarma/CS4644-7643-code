import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    flash_attn = False,
    cond_channels=1 # 19 for semantic, 1 for edge aware 
)
diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    sampling_timesteps=250,
    objective='pred_noise',
    min_snr_loss_weight=True,
    min_snr_gamma=5,
    offset_noise_strength=0.0,
    adaptive=False, # True for spatially adaptive noise
    masked_loss=False, # True for spatially adaptive noise
)

trainer = Trainer(
    diffusion,
    "/home/hice1/avarma49/scratch/ffhq-128/",
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 200000,
    save_and_sample_every=1000, # 1000 is deafult
    gradient_accumulate_every = 1,
    ema_decay = 0.995, # exponential moving average decay
    amp = True,  # mixed precision
    calculate_fid = True,
    num_fid_samples=5000,
    # results_folder="/home/hice1/avarma49/scratch/results/results128_edgeaware_noise",
    results_folder="/home/hice1/avarma49/scratch/results/results128_conditional_semantic_v2",
    # results_folder="/home/hice1/avarma49/scratch/results/results128_edgeaware_masked_loss",
    # results_folder="/home/hice1/avarma49/scratch/results/results128_default",
    mask_folder = "/home/hice1/avarma49/scratch/ffhq-128-masks", # only on for semantic - going to be automatically ignored for edge_aware
    # adaptive_mask_type = "edge_aware", # only for adaptive noise
    conditional_mask_type = "semantic",
    # masked_loss_type = "semantic"
)

# resume from checkpoint
trainer.load(175)
trainer.train()