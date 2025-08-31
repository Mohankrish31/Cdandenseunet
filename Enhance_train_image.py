import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')   # change if needed
from models.cdan_denseunet import CDANDenseUNet  # replace with your model class

# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/test/low"   # input low-light images
output_dir = "/content/outputs/test_enhanced"     # output folder
model_path = "/content/saved_model/cdan_denseunet.pth"  # trained weights

os.makedirs(output_dir, exist_ok=True)

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDANDenseUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# ------------------- Transforms -------------------
to_tensor = ToTensor()      # converts [0–255] -> [0–1], shape [C,H,W]
to_pil = ToPILImage()       # converts back to PIL [H,W,C] in uint8

# ------------------- Inference Loop -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load and preprocess
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        inp = to_tensor(img).unsqueeze(0).to(device)  # shape [1,C,H,W]

        # Forward pass
        out = model(inp)

        # Clamp values to [0,1] (important!)
        out = torch.clamp(out.squeeze(0), 0, 1)

        # Convert to PIL
        enhanced_img = to_pil(out.cpu())

        # Save
        enhanced_img.save(os.path.join(output_dir, fname))

        # Debug info
        avg_rgb = out.mean().item()
        print(f"{fname} -> Avg RGB (0–1): {avg_rgb:.4f} | Saved ✅")
