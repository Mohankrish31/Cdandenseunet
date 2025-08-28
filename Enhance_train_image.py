import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_pil_image
import cv2

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

# ------------------- CLAHE for contrast enhancement -------------------
def clahe_enhance(img_np):
    img_clahe = np.zeros_like(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for c in range(3):
        channel = (img_np[:,:,c]*255).astype(np.uint8)
        channel = clahe.apply(channel)
        img_clahe[:,:,c] = channel / 255.0
    return img_clahe

# ------------------- Gray World Color Correction -------------------
def gray_world_correction(img_np):
    mean_rgb = img_np.mean(axis=(0,1))
    mean_gray = mean_rgb.mean()
    scale = mean_gray / mean_rgb
    img_corrected = img_np * scale
    img_corrected = np.clip(img_corrected, 0, 1)
    return img_corrected

# ------------------- Noise Reduction -------------------
def denoise_image(img_pil):
    # Mild Gaussian blur
    return img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))

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

        # Convert to NumPy for post-processing
        out_np = out.permute(1,2,0).numpy()

        # ------------------- Post-processing -------------------
        out_np = clahe_enhance(out_np)            # Contrast enhancement
        out_np = gray_world_correction(out_np)    # Color correction
        out_pil = to_pil_image(torch.tensor(out_np).permute(2,0,1).float())

        out_pil = denoise_image(out_pil)          # Noise reduction

        # Save output
        save_path = os.path.join(output_dir, fname)
        out_pil.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")

print("\n✅ All images processed successfully!")
