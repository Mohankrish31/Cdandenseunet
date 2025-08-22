import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
# -------- Add model path --------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"
# -------- Create output dir --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Model Architecture --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
# -------- Load weights --------
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
# -------- Preprocessing (⚠️ Must match training) --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 🔥 If you trained with normalization, UNCOMMENT this:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()
# -------- Enhance images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            out = model(inp)   # [1,3,224,224]
            out = out.squeeze().detach().cpu()
            # 🔥 Always rescale safely
            if out.min() < 0:  
                out = (out + 1) / 2  # if model trained with tanh → [-1,1]
            out = out.clamp(0, 1)
            print(f"{fname} → Output range: {out.min().item():.4f} to {out.max().item():.4f}")
            out_img = to_pil(out)
            out_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved (train): {fname}")
print("🎉 All training images processed and saved to:", output_dir)
