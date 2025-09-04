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

# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------- Consistent Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load image (CONSISTENT with Program 1)
        img_path = os.path.join(input_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")  # ✅ Consistent loading
        
        # Get original size for resizing back
        original_size = pil_img.size  # (width, height)
        
        # Forward pass through model
        inp = transform(pil_img).unsqueeze(0).to(device)
        out = model(inp).squeeze(0)  # [3, H, W]
        
        # Clamp to valid range
        out = out.clamp(0, 1)  
        
        # Convert to numpy and scale to 0-255
        out_np = out.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        out_uint8 = (out_np * 255).astype(np.uint8)
        
        # Resize back to original dimensions
        out_resized = cv2.resize(out_uint8, original_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert RGB to BGR for OpenCV saving
        out_bgr = cv2.cvtColor(out_resized, cv2.COLOR_RGB2BGR)
        
        # Save result
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, out_bgr)

        print(f"✅ Saved {save_path}")

print("🎯 All images processed successfully!")
