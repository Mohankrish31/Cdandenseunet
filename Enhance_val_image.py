import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
# -------- Add model path --------
sys.path.append('/content/CbamDenseUnet')
from models.cdan_denseunet import CDANDenseUNet   # ✅ use your architecture class
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/val/low"  # Low-light training images
output_dir = "/content/outputs/val_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"   # ✅ weights file
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Recreate model architecture --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
# -------- Load weights --------
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)   # ✅ load weights into model
model.eval()
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match training resolution
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()
# -------- Enhance and save training images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            out = model(inp).squeeze().cpu().clamp(0, 1)
            out_img = to_pil(out)
            out_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved (train): {fname}")
print("🎉 All valid images processed and saved to:", output_dir)
