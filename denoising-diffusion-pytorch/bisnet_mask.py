import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Clone BiSeNet repo or download pretrained model weights first
# https://github.com/zllrunning/face-parsing.PyTorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BiSeNet model (assuming you have the model code and weights)
from model import BiSeNet  # you need to have the model.py from the repo

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.to(device)
net.load_state_dict(torch.load('79999_iter.pth', map_location=device))
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_mask(image_path, save_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img_resized = cv2.resize(img, (512, 512))
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(img_tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    # Resize back to original size
    parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Save mask
    Image.fromarray(parsing).save(save_path)

if __name__ == '__main__':
    image_folder = 'dataset/images'
    mask_folder = 'dataset/masks'
    os.makedirs(mask_folder, exist_ok=True)

    image_list = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    for img_name in image_list:
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name.replace('.jpg', '.png'))

        generate_mask(img_path, mask_path)
        print(f"Saved mask for {img_name}")
