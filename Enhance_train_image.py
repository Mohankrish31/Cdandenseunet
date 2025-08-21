import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
# -------- Add model path --------
sys.path.append('/content/CbamDenseUnet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"      # Low-light test images
output_dir = "/content/outputs/train_enhanced"       # Folder to save enhanced images
model_path = "/content/models/cdan_denseunet.pt"
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Load Model --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
# -------- Transform --------
to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()
# -------- Enhancement Loop --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            # Model Inference
            inp = to_tensor(img).unsqueeze(0).to(device)
            out = model(inp).squeeze().cpu().clamp(0, 1)
            # Convert to PIL Image
            out_img = to_pil(out)
            out_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved: {fname}")
print("🎉 All train images processed and saved to:", output_dir)
