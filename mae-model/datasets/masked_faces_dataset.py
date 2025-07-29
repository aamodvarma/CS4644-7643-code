import os, random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MaskedFacesDataset(Dataset):
    def __init__(self, image_dir, mask_root, filenames, mask_variants, threshold=0.5, seed=123):
        self.image_dir = image_dir
        self.mask_root = mask_root
        self.mask_variants = mask_variants
        self.threshold = threshold

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))

        self.filenames = []
        for fname in filenames:
            has_all_masks = all(os.path.exists(os.path.join(mask_root, variant, fname)) for variant in mask_variants)
            if has_all_masks:
                self.filenames.append(fname)

        print(f"Using {len(self.filenames)} images with complete masks.")

        # === Fix one mask variant per image ===
        random.seed(seed)
        self.mask_assignment = {
            fname: random.choice(self.mask_variants) for fname in self.filenames
        }

        print(f"Originally {len(filenames)} images, filtered to {len(self.filenames)} with complete masks.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # === Load image ===
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        img = self.resize(img)
        img_tensor = self.to_tensor(img)

        # === Load fixed assigned mask variant ===
        variant = self.mask_assignment[fname]
        mask_path = os.path.join(self.mask_root, variant, fname)
        mask = Image.open(mask_path).convert('L')
        mask_arr = np.array(self.resize(mask))
        mask_bin = (mask_arr > 127).astype(np.uint8)

        # === Convert to patch-level mask ===
        patch_mask = mask_bin.reshape(14, 16, 14, 16).mean(axis=(1, 3))
        patch_mask = (patch_mask > self.threshold).astype(np.uint8)
        patch_mask_tensor = torch.tensor(patch_mask.flatten(), dtype=torch.bool)

        return img_tensor, patch_mask_tensor
