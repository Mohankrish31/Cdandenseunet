import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# Add model folder to path
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet_isp_weights.pth"
os.makedirs(output_dir, exist_ok=True)
# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- Load Model -------------------
try:
    # ⚠️ Make sure you use the SAME output_range as training
    model = CDANDenseUNet(in_channels=3, base_channels=32, output_range="01").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Model file not found at {model_path}")
    sys.exit()
# ------------------- Preprocessing -------------------
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor
# ------------------- Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_dir, fname)
        inp = preprocess_image(img_path).to(device)
        out = model(inp).squeeze(0).cpu()
        # ✅ Handle both output ranges
        if getattr(model, "output_range", "01") == "11":
            out = (out + 1) / 2  # map [-1,1] -> [0,1]
        out = torch.clamp(out, 0, 1)
        print(fname, "-> min:", out.min().item(), "max:", out.max().item())
        if out.shape[0] == 1:
            out = out.repeat(3, 1, 1)
        out_img = to_pil_image(out)
        save_path = os.path.join(output_dir, fname)
        out_img.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")
print("\n✅ All images processed successfully!")
