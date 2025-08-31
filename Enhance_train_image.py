import os
import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')  # change if needed
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/test/low"   # folder with low-light input images
output_dir = "/content/outputs/test_enhanced"    # folder to save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pth"  # trained weights
os.makedirs(output_dir, exist_ok=True)
# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDANDenseUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,   # MUST match your trained model
    growth_rate=12,
    output_range="01"   # outputs in [0,1]
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully.")
# ------------------- Transforms -------------------
to_tensor = ToTensor()      # converts PIL [0-255] -> tensor [0-1]
to_pil = ToPILImage()       # converts tensor [0-1] -> PIL [0-255]
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
        # Clamp output to [0,1]
        out = torch.clamp(out.squeeze(0), 0, 1)
        # Convert to PIL and save
        enhanced_img = to_pil(out.cpu())
        enhanced_img.save(os.path.join(output_dir, fname))
        # Debug info
        avg_rgb = out.mean().item()
        print(f"{fname} -> Avg RGB (0–1): {avg_rgb:.4f} | Saved ✅")
