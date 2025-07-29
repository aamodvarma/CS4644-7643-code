import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from models_mae import mae_vit_base_patch16

# === Config ===
img_dir = 'data/inference/images'
mask_dirs = [
    'masks_edge_25', 'masks_edge_50',
    'masks_structured-all', 'masks_structured-e', 'masks_structured-nm'
]
mask_dirs = [os.path.join('data/inference', m) for m in mask_dirs]
output_root = 'outputs_structured_finetune'
checkpoint_path = 'checkpoints_structured_finetune/checkpoint-9.pth'

threshold = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load finetuned model ===
model = mae_vit_base_patch16()
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()

# === Preprocessing ===
to_tensor = transforms.ToTensor()
resize_224 = transforms.Resize((224, 224))
image_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])

# === Masking Utility ===
def apply_patch_mask_to_image(image_tensor, patch_mask, patch_size=16):
    image = image_tensor.clone()
    patch_mask = patch_mask.reshape(14, 14)
    for i in range(14):
        for j in range(14):
            if patch_mask[i, j]:
                image[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0.0
    return image

# === Inference loop ===
mask_coverage_stats = {}

for mask_dir in mask_dirs:
    mask_name = os.path.basename(mask_dir)
    print(f'Running inference with {mask_name} | threshold={threshold}')

    output_dir = os.path.join(output_root, f"{mask_name}_thresh{int(threshold*100)}")
    output_dir_masked = os.path.join(output_dir, "masked")
    output_dir_combined = os.path.join(output_dir, "nonmask+recon")

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir_masked, exist_ok=True)
    os.makedirs(output_dir_combined, exist_ok=True)

    total_masked = 0
    total_images = 0

    for fname in tqdm(image_filenames):
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert('RGB')
        img = resize_224(img)
        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            print(f"Skipping {fname} — mask missing in {mask_name}")
            continue

        mask_img = Image.open(mask_path).convert('L')
        mask_arr = np.array(resize_224(mask_img))
        mask_arr = (mask_arr > 127).astype(np.uint8)

        patch_mask = mask_arr.reshape(14, 16, 14, 16).mean(axis=(1, 3))
        patch_mask = (patch_mask > threshold).astype(np.uint8)
        patch_mask_tensor = torch.tensor(patch_mask).reshape(1, -1).bool().to(device)

        if patch_mask_tensor.sum() == 0 or patch_mask_tensor.sum() >= 196:
            print(f"Skipping {fname} (mask sum={patch_mask_tensor.sum().item()})")
            continue

        total_masked += patch_mask_tensor.sum().item()
        total_images += 1

        with torch.no_grad():
            loss, pred, _ = model(img_tensor, mask=patch_mask_tensor)

        recon = model.unpatchify(pred).clamp(0, 1)

        # === Save masked image
        masked_tensor = apply_patch_mask_to_image(img_tensor, patch_mask_tensor[0])
        masked_img = transforms.ToPILImage()(masked_tensor.squeeze(0).cpu())
        masked_img.save(os.path.join(output_dir_masked, fname))

        # === Save combined: original unmasked + reconstructed masked
        patch_mask_2d = patch_mask_tensor.reshape(1, 1, 14, 14).float()
        mask_unpatched = F.interpolate(patch_mask_2d, size=(224, 224), mode='nearest')
        combined = img_tensor * (1 - mask_unpatched) + recon * mask_unpatched
        combined_img = transforms.ToPILImage()(combined[0].cpu())
        combined_img.save(os.path.join(output_dir_combined, fname))

    avg_pct = (total_masked / (total_images * 196)) * 100 if total_images > 0 else 0
    mask_coverage_stats[mask_name] = avg_pct

# === Save summary stats
with open(os.path.join(output_root, "avg_patch_mask_coverage.txt"), "w") as f:
    for mask_name, pct in mask_coverage_stats.items():
        f.write(f"{mask_name} | threshold={threshold}: avg patch masked = {pct:.2f}%\n")

print("Inference complete.")
