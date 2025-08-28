import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')  # change to your path
from models.cdan_denseunet import CDANDenseUNet  # replace if using custom model
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"   # low-light images
output_dir = "/content/outputs/train_enhanced"    # save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pth"  # trained weights
os.makedirs(output_dir, exist_ok=True)
# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- Load Model -------------------
model = CDANDenseUNet(in_channels=3, base_channels=32, output_range="01").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")
# ------------------- Preprocessing -------------------
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0  # [0,255] → [0,1]
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor
# ------------------- Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_dir, fname)
        inp = preprocess_image(img_path).to(device)
        # Run the model
        out = model(inp).squeeze(0).cpu()  # [C,H,W]
        out = torch.clamp(out, 0, 1)      # keep in [0,1]
        # Convert to 0-255 for saving
        out_img = (out * 255.0).byte()
        out_pil = to_pil_image(out_img)
        save_path = os.path.join(output_dir, fname)
        out_pil.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")
print("\n✅ All images processed successfully!")
