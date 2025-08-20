import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/test/low"   # Low-light test images
output_dir = "/content/drive/MyDrive/Colon_Enhanced/test_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model and weights --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
# -------- Preprocessing (same as training) --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # [0,1], no Normalize since training didn’t use it
])
# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            # Model inference
            outs = model(inp).cpu().detach()  # [B, 3, H, W]
            # Debug: check raw output range
            print(f"[{fname}] Raw min:", outs.min().item(), "Raw max:", outs.max().item())
            # Normalize to [0,1] for saving
            outs = (outs - outs.min()) / (outs.max() - outs.min() + 1e-8)
            # Debug: check normalized range
            print(f"[{fname}] Normalized min:", outs.min().item(), "max:", outs.max().item())
            # Process each output in batch
            for i in range(outs.size(0)):
                out_img = to_pil_image(outs[i])  # Tensor → PIL
                save_path = os.path.join(output_dir, f"enhanced_{fname}")
                out_img.save(save_path)
                print(f"✅ Enhanced & saved (test): {save_path}")
print("🎉 All test images processed and saved to:", output_dir)
