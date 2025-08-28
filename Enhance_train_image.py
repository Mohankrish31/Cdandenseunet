import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')  # Change to your path
from models.cdan_denseunet import CDANDenseUNet  # Your custom model
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"  # Low-light images
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pth"
os.makedirs(output_dir, exist_ok=True)

# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Preprocessing Function -------------------
def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image: RGB -> [0-1] float tensor"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor

# ------------------- Load Model -------------------
model = CDANDenseUNet(in_channels=3, base_channels=32, output_range="01").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# ------------------- Post-processing (Gamma Correction) -------------------
def linear_to_srgb(image):
    """Convert linear [0-1] float to sRGB [0-255] uint8."""
    srgb_image = torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * (image ** (1/2.4)) - 0.055
    )
    srgb_image = torch.clamp(srgb_image * 255.0, 0, 255).to(torch.uint8)
    return srgb_image

# ------------------- Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_dir, fname)

        # --- Step 1: Preprocess ---
        inp = preprocess_image(img_path).to(device)

        # --- Step 2: Model Inference ---
        out = model(inp).squeeze(0).cpu()
        out = torch.clamp(out, 0, 1)

        # --- Step 3: Min-max normalization across all channels (avoid pink tint) ---
        min_val = out.min()
        max_val = out.max()
        out = (out - min_val) / (max_val - min_val + 1e-8)

        # --- Step 4: Compute average RGB value ---
        avg_rgb = out.mean().item()  # single scalar for overall brightness
        print(f"{fname} -> Average RGB: {avg_rgb:.4f}")

        # --- Step 5: Post-processing (sRGB conversion) ---
        enhanced_img = linear_to_srgb(out)

        # --- Step 6: Save Image ---
        out_pil = to_pil_image(enhanced_img)
        save_path = os.path.join(output_dir, fname)
        out_pil.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")

print("\n✅ All images processed successfully!")
