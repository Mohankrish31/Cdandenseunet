import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')   # 👈 your model folder
from models.cdan_denseunet import CDANDenseUNet

# -------- Paths --------
input_dir = "/content/cvccolondbsplit/val/low"   # Low-light training images
output_dir = "/content/outputs/val_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"   # 👈 saved weights

# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)

# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load model and weights --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # keep [0,1] range
])

# -------- Enhance and save training images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            
            # Preprocess
            inp = transform(img).unsqueeze(0).to(device)

            # Model forward
            out = model(inp).squeeze().cpu()

            # Clamp to valid [0,1] range
            out = out.clamp(0, 1)

            # Convert to PIL
            out_img = to_pil_image(out)

            # Save enhanced output
            save_path = os.path.join(output_dir, f"enhanced_{fname}")
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {save_path}")

print("🎉 All validation images processed and saved to:", output_dir)
