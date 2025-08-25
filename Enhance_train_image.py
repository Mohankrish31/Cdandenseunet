import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# Add model folder to path
sys.path.append('/content/CdanDenseUNet')  # Adjust to your folder
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"       # Low-light images
output_dir = "/content/outputs/train_enhanced"        # Enhanced images save path
model_path = "/content/saved_model/cdan_denseunet_isp_weights.pth"  # Trained weights
os.makedirs(output_dir, exist_ok=True)
# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- Load Model -------------------
try:
    # Initialize model
    model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Model file not found at {model_path}")
    sys.exit()
# ------------------- Preprocessing -------------------
def preprocess_image(img_path, target_size=(224, 224)):
    """Load image, resize and normalize to [0,1]"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0  # normalize
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)  # HWC → CHW + batch
    return img_tensor
# ------------------- Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_dir, fname)
        inp = preprocess_image(img_path).to(device)
        # Forward pass
        out = model(inp)
        # Clamp output to [0,1] and remove batch dimension
        out = torch.clamp(out.squeeze(0), 0, 1).cpu()
        # If single-channel, convert to 3-channel
        if out.shape[0] == 1:
            out = out.repeat(3, 1, 1)
        # Convert to PIL image and save
        out_img = to_pil_image(out)
        save_path = os.path.join(output_dir, fname)
        out_img.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")
print("\n✅ All images processed successfully!")
