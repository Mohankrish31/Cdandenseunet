import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light training images
output_dir = "/content/outputs/train_enhanced"     # Enhanced images
model_path = "/content/saved_model/cdan_denseunet.pt"  # Best model (full)
os.makedirs(output_dir, exist_ok=True)
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load full model --------
model = torch.load(model_path, map_location=device)
model = model.to(device).eval()
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # must match training resolution
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()
# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)   # [1,3,H,W]
            out = model(inp).squeeze().cpu().clamp(0, 1)   # [3,H,W]
            out_img = to_pil(out)
            out_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved: {fname}")
print("🎉 All training images processed and saved to:", output_dir)
