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
    output_range="01"
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------- Transforms (WITH resize 224x224) -------------------
resize_dim = (224, 224)
transform = transforms.Compose([
    transforms.Resize(resize_dim),  # Resize before tensor conversion
    transforms.ToTensor()
])

# ------------------- Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load image in RGB
        img_path = os.path.join(input_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")

        # Resize + convert to tensor
        inp = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # Forward pass
        out = model(inp).squeeze(0)  # [3, 224, 224]

        # Clamp to valid range
        out = out.clamp(0, 1)

        # Convert to numpy (H, W, 3)
        out_np = out.permute(1, 2, 0).cpu().numpy()
        out_uint8 = (out_np * 255).astype(np.uint8)

        # Convert RGB → BGR for saving
        out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)

        # Save result
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, out_bgr)

        print(f"✅ Saved {save_path}")

print("🎯 All images processed successfully at 224x224!")
