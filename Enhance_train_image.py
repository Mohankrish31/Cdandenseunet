import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light test images
output_dir = "/content/drive/MyDrive/Colon_Enhanced/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model and weights --------
model =  CDANDenseUNet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load(cdan_model_path, map_location=device))
model.eval()
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()
# -------- Enhance and save testing images (NO POSTPROCESSING) --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            # Model inference
            out = model(inp).squeeze().cpu().clamp(0, 1)
            out_img = to_pil(out)
            # Save result
            out_cv = np.array(out_img)
            final_img = Image.fromarray(out_cv)
            final_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved (train): {fname}")
print("🎉 All train images processed and saved to:", output_dir) 
