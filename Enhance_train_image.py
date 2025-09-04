import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

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
    output_range="01"
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match training
    transforms.ToTensor()
])
to_pil = ToPILImage()

# ------------------- Inference Loop -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load image
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)

        # Forward pass
        out = model(inp).squeeze(0)  # [3,H,W]

        # --------- White balance / channel correction ---------
        # scale each channel slightly to match mean brightness
        channel_means = out.mean(dim=(1,2))
        mean_gray = channel_means.mean()
        correction = (mean_gray / (channel_means + 1e-6)).clamp(0.8, 1.2)  # ±20%
        out = out * correction.view(3,1,1)
        out = torch.clamp(out, 0, 1)

        # --------- Optional min-max normalization per image ---------
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)

        # Convert tensor -> numpy -> BGR
        out_np = out.permute(1, 2, 0).cpu().numpy()
        out_uint8 = (out_np * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)

        # Save image
        cv2.imwrite(os.path.join(output_dir, fname), out_bgr)

        # Debug info
        r_mean, g_mean, b_mean = out_np[...,0].mean(), out_np[...,1].mean(), out_np[...,2].mean()
        print(f"{fname} -> R:{r_mean:.3f} G:{g_mean:.3f} B:{b_mean:.3f} | Saved ✅")

print("✅ All images enhanced and saved successfully!")
