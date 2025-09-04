import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet

# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pth"
os.makedirs(output_dir, exist_ok=True)

# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Load Model -------------------
model = CDANDenseUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,
    growth_rate=12,
    output_range="01"   # ensures outputs in [0,1]
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------- Inference (No Post-processing) -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load original image
        img_path = os.path.join(input_dir, fname)
        orig_bgr = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(orig_rgb)

        # Forward pass through model
        inp = transform(pil_img).unsqueeze(0).to(device)
        out = model(inp).squeeze(0)  # [3,H,W]

        # Clamp & convert to numpy
        out = out.clamp(0, 1)  
        out_np = out.permute(1, 2, 0).cpu().numpy()

        # Resize back to original size
        out_resized = cv2.resize((out_np * 255).astype(np.uint8),
                                 (orig_rgb.shape[1], orig_rgb.shape[0]))

        # Save result (RGB → BGR for OpenCV)
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(out_resized, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved {save_path} (direct model output, no post-processing)")

print("🎯 All images enhanced and saved directly from model output!")
