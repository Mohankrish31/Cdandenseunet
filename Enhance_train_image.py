import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')  # change to your path
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"     # folder with low-light input images
output_dir = "/content/outputs/train_enhanced"       # folder to save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pth"  # trained weights
os.makedirs(output_dir, exist_ok=True)
# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- Load Model -------------------
model = CDANDenseUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,      # MUST match your trained model
    growth_rate=12,
    output_range="01"      # outputs in [0,1]
)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model state dict: {e}")
    sys.exit(1)
model.to(device)
model.eval()
# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match training
    transforms.ToTensor()
])
to_pil = ToPILImage()
# ------------------- Normalization helper -------------------
def normalize_and_balance(img_np):
    """
    img_np: numpy image in [0,1], shape (H, W, 3), RGB
    - Normalizes to min=0, max=1
    - Adjusts per-channel mean to ~0.5
    """
    # Step 1: min-max normalize
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    # Step 2: adjust channel mean toward 0.5
    for c in range(3):
        mean_c = img_np[..., c].mean()
        if mean_c > 1e-5:
            scale = 0.5 / (mean_c + 1e-8)
            img_np[..., c] = np.clip(img_np[..., c] * scale, 0, 1)
    return img_np
# ------------------- Inference Loop -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        # Load and preprocess
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)  # shape [1,3,224,224]
        # Forward pass
        out = model(inp).squeeze(0)  # shape [3,H,W]
        # ---------- Channel-wise mean correction ----------
        # Normalize tensor to [0,1]
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        channel_means = out.mean(dim=(1,2))
        mean_gray = channel_means.mean()
        correction = (mean_gray / (channel_means + 1e-6)).clamp(0.9, 1.1)  # ±10% only
        out = out * correction.view(3,1,1)
        out = torch.clamp(out, 0, 1)
        # ---------- Convert tensor → numpy ----------
        out_np = out.permute(1, 2, 0).cpu().numpy()  # shape (H, W, 3)
        # Apply additional normalization & channel balancing if needed
        out_np = normalize_and_balance(out_np)
        # Save result
        out_uint8 = (out_np * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, fname), out_bgr)
        # Debug info per channel
        r_mean, g_mean, b_mean = out_np[..., 0].mean(), out_np[..., 1].mean(), out_np[..., 2].mean()
        print(f"{fname} -> R_mean: {r_mean:.4f}, G_mean: {g_mean:.4f}, B_mean: {b_mean:.4f} | Saved ✅")
print("✅ All images enhanced, channel-corrected, normalized, and saved successfully!")
