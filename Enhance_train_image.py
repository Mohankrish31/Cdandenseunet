import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')   # 👈 adjust your model folder if needed
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # low-light test images
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"   # saved weights
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model and weights --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # keep values in [0,1]
])
# -------- Enhance and save testing images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            # Preprocess
            inp = transform(img).unsqueeze(0).to(device)
            # Model forward
            out = model(inp)
            # Remove batch dim -> [3,H,W]
            out = out.squeeze(0).cpu()
            # Debug: check output range
            print(fname, "-> min:", out.min().item(), "max:", out.max().item())
            # If model outputs in [-1,1], rescale to [0,1]
            if out.min() < 0:
                out = (out + 1) / 2.0
            # Clamp to valid [0,1]
            out = out.clamp(0, 1)
            # Convert to PIL
            out_img = to_pil_image(out)
            # Save enhanced output
            save_path = os.path.join(output_dir, f"enhanced_{fname}")
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {save_path}")
print("🎉 All training images processed and saved to:", output_dir)
