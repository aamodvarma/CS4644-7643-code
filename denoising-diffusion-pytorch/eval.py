import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.kid_evaluation import KIDEvaluation

model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    flash_attn = False,
    # cond_channels=1 # 19 for semantic, 1 for edge aware 
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
    adaptive=True, # True for spatially adaptive noise
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
    # results_folder="/home/hice1/avarma49/scratch/results/results128_conditional_edgeaware_mask",
    results_folder="/home/hice1/avarma49/scratch/results/results128_edgeaware_noise",
    # results_folder="/home/hice1/avarma49/scratch/results/results128_default",
    mask_folder = "/home/hice1/avarma49/scratch/ffhq-128-masks", # only on for semantic - going to be automatically ignored for edge_aware
    adaptive_mask_type = "edge_aware", # only for adaptive noise
    # conditional_mask_type = "edge_aware",
    # masked_loss_type = "edge_aware"
)

# resume from checkpoint
trainer.load(175)
kid_scorer = KIDEvaluation(
    batch_size=trainer.batch_size,
    dl=trainer.dl,
    sampler=trainer.ema.ema_model,
    channels=trainer.channels,
    accelerator=trainer.accelerator,
    device=trainer.device,
    num_kid_samples=5000,
    inception_block_idx=2048,
    conditional_mask_type=trainer.conditional_mask_type 
)

kid_value = kid_scorer.kid_score()
print("KID:", kid_value)
