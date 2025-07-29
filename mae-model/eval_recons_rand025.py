import os
import torch
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.nn.functional import normalize
from tqdm import tqdm
from PIL import Image
import numpy as np
from facenet_pytorch import InceptionResnetV1
import lpips

# === CONFIG ===
image_dir = "data/inference/images"
output_root = "outputs_rand025"
nonmask_folder = "nonmask+recon"
summary_path = os.path.join(output_root, "summary_metrics_rand025.txt")

# mask folders to evaluate
structured_folders = [
    "masks_structured-all_thresh50",
    "masks_structured-e_thresh50",
    "masks_structured-nm_thresh50",
    "masks_edge_25_thresh50",
    "masks_edge_50_thresh50"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Models ===
fid = FrechetInceptionDistance(feature=2048).to(device)
arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)

# === Transforms ===
arcface_transform = T.Compose([T.Resize((160, 160)), T.ToTensor()])
fid_transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])
base_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

# === Header ===
header = "folder,FID,ArcFace_CosSim,SSIM,PSNR,LPIPS\n"
if not os.path.exists(summary_path):
    with open(summary_path, "w") as f:
        f.write(header)

# === Evaluate each folder ===
for folder in structured_folders:
    print(f"\n🔍 Evaluating {folder}")

    orig_paths = sorted([
        os.path.join(image_dir, fn)
        for fn in os.listdir(image_dir)
        if fn.endswith(('.png', '.jpg'))
    ])
    recon_dir = os.path.join(output_root, folder, nonmask_folder)
    recon_paths = sorted([
        os.path.join(recon_dir, fn)
        for fn in os.listdir(recon_dir)
        if fn.endswith(('.png', '.jpg'))
    ])

    # Ensure consistent matching
    common = set(os.path.basename(p) for p in orig_paths).intersection(
        os.path.basename(p) for p in recon_paths
    )
    orig_paths = [p for p in orig_paths if os.path.basename(p) in common]
    recon_paths = [p for p in recon_paths if os.path.basename(p) in common]

    if len(orig_paths) == 0:
        print(f"No overlapping images found for {folder}")
        continue

    # === FID ===
    fid.reset()
    for p in tqdm(orig_paths, desc=f"{folder} FID (real)"):
        img = fid_transform(default_loader(p)).mul(255).clamp(0, 255).to(torch.uint8).unsqueeze(0).to(device)
        fid.update(img, real=True)
    for p in tqdm(recon_paths, desc=f"{folder} FID (fake)"):
        img = fid_transform(default_loader(p)).mul(255).clamp(0, 255).to(torch.uint8).unsqueeze(0).to(device)
        fid.update(img, real=False)
    fid_score = fid.compute().item()

    # === ArcFace Cosine Similarity ===
    cos_sims = []
    for p1, p2 in tqdm(zip(orig_paths, recon_paths), total=len(orig_paths), desc=f"{folder} ArcFace CosSim"):
        t1 = arcface_transform(default_loader(p1)).unsqueeze(0).to(device)
        t2 = arcface_transform(default_loader(p2)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb1 = normalize(arcface(t1))
            emb2 = normalize(arcface(t2))
        cos_sims.append((emb1 @ emb2.T).item())
    arcface_score = np.mean(cos_sims)

    # === SSIM / PSNR / LPIPS ===
    ssim_vals, psnr_vals, lpips_vals = [], [], []
    for p1, p2 in tqdm(zip(orig_paths, recon_paths), total=len(orig_paths), desc=f"{folder} SSIM/PSNR/LPIPS"):
        t1 = base_transform(default_loader(p1)).unsqueeze(0).to(device)
        t2 = base_transform(default_loader(p2)).unsqueeze(0).to(device)
        ssim_vals.append(ssim_metric(t1, t2).item())
        psnr_vals.append(psnr_metric(t1, t2).item())
        lpips_vals.append(lpips_metric(t1, t2).item())
    ssim_score = np.mean(ssim_vals)
    psnr_score = np.mean(psnr_vals)
    lpips_score = np.mean(lpips_vals)

    # === Write to file ===
    with open(summary_path, "a") as f:
        f.write(f"{folder},{fid_score:.4f},{arcface_score:.4f},{ssim_score:.4f},{psnr_score:.4f},{lpips_score:.4f}\n")

    print(f"{folder} | FID={fid_score:.4f} | ArcFace={arcface_score:.4f} | SSIM={ssim_score:.4f} | PSNR={psnr_score:.4f} | LPIPS={lpips_score:.4f}")
