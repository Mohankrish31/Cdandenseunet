import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pth"
os.makedirs(output_dir, exist_ok=True)
# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- Load Model -------------------
try:
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
        out = torch.clamp(out, 0, 1)
        # Save output as image
        out_pil = to_pil_image(out)
        save_path = os.path.join(output_dir, fname)
        out_pil.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")
print("\n✅ All images processed successfully!")
