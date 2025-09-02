import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')  # change to your path
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"     # folder with low-light input images
output_dir = "/content/outputs/train_enhanced"      # folder to save enhanced images
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
    sys.exit(1)  # Exit if model can't be loaded
model.to(device)
model.eval()
# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match training
    transforms.ToTensor()
])
to_pil = ToPILImage()  # converts tensor [0-1] -> PIL [0-255]
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
        out = model(inp).squeeze(0).cpu()
        # ------------------- Per-channel contrast stretching -------------------
        out = out.clone()  # make a copy
        for c in range(3):  # R, G, B channels
            min_val = out[c].min()
            max_val = out[c].max()
            if max_val - min_val > 1e-5:  # avoid division by zero
                out[c] = (out[c] - min_val) / (max_val - min_val)
        out = out.clamp(0, 1)  # ensure range [0,1]
        # Convert to PIL and save
        enhanced_img = to_pil(out)
        enhanced_img.save(os.path.join(output_dir, fname))
        # Debug info per channel
        r_mean, g_mean, b_mean = out[0].mean().item(), out[1].mean().item(), out[2].mean().item()
        print(
            f"{fname} -> R_mean: {r_mean:.4f}, G_mean: {g_mean:.4f}, B_mean: {b_mean:.4f} | Saved ✅"
        )
print("✅ All images enhanced and saved successfully!")
