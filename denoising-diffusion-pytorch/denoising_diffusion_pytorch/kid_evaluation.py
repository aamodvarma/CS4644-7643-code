import math
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import create_edge_aware_mask
import os
import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm

def polynomial_kernel(x, y):
    d = x.shape[1]
    return (x @ y.T / d + 1) ** 3

def mmd_unbiased(x, y):
    n = x.shape[0]
    m = y.shape[0]
    k_xx = polynomial_kernel(x, x)
    k_yy = polynomial_kernel(y, y)
    k_xy = polynomial_kernel(x, y)
    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2 * sum_xy

class KIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        device="cuda",
        num_kid_samples=50000,
        inception_block_idx=2048,
        conditional_mask_type=None
    ):
        self.batch_size = batch_size
        self.n_samples = num_kid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.conditional_mask_type = conditional_mask_type
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    @torch.inference_mode()
    def kid_score(self):
        self.sampler.eval()
        num_batches = math.ceil(self.n_samples / self.batch_size)
        real_features_list, fake_features_list = [], []

        self.print_fn(
            f"Computing Inception features for {self.n_samples} real and generated samples."
        )

        for _ in tqdm(range(num_batches)):
            data = next(self.dl)
            if self.conditional_mask_type == "semantic":
                real_samples, mask = data
            elif self.conditional_mask_type == "edge_aware":
                real_samples = data
                mask = create_edge_aware_mask(real_samples)
            else:
                if isinstance(data, list):
                    real_samples, _ = data
                else:
                    real_samples = data
                mask = None

            real_samples = real_samples.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            fake_samples = (
                self.sampler.sample(batch_size=real_samples.size(0), cond=mask)
                if mask is not None
                else self.sampler.sample(batch_size=real_samples.size(0))
            )

            real_features = self.calculate_inception_features(real_samples)
            fake_features = self.calculate_inception_features(fake_samples)

            real_features_list.append(real_features)
            fake_features_list.append(fake_features)

        real_features = torch.cat(real_features_list, dim=0)
        fake_features = torch.cat(fake_features_list, dim=0)

        kid = mmd_unbiased(real_features, fake_features)
        return kid.item()
