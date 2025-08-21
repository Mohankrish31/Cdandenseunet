import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# -------- Add model path --------
sys.path.append('/content/CbamDenseUnet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/test/low"      # Low-light test images
output_dir = "/content/outputs/test_enhanced"       # Folder to save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pt"   # full saved model (.pt)
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Load Full Model --------
print("🔹 Loading full model...")
model = torch.load(model_path, map_location=device)
model = model.to(device).eval()
print("✅ Loaded full model successfully!")
# -------- Transform --------
to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),   # 👈 change/remove this if you want original resolution
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()
# -------- Enhancement Loop --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            # Preprocess
            inp = to_tensor(img).unsqueeze(0).to(device)
            # Model forward
            out = model(inp).squeeze().cpu()
            # Rescale if model outputs [-1,1]
            if out.min() < 0:
                out = (out + 1) / 2.0
            out = out.clamp(0, 1)
            # Convert to PIL
            out_img = to_pil(out)
            # Save
            save_path = os.path.join(output_dir, f"enhanced_{fname}")
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {save_path}")
print("🎉 All test images processed and saved to:", output_dir)
