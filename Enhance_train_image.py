import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

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
model = CDANDenseUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,
    growth_rate=12,
    output_range="01"
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------- Inference + Enhancement -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load original image
        img_path = os.path.join(input_dir, fname)
        orig_bgr = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(orig_rgb)

        # Forward pass through model
        inp = transform(pil_img).unsqueeze(0).to(device)
        out = model(inp).squeeze(0)  # [3,H,W]

        # Normalize output to [0,1]
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        out_np = out.permute(1, 2, 0).cpu().numpy()
        out_uint8 = (out_np * 255).astype(np.uint8)

        # ------------------- Use model as luminance (L channel) -------------------
        model_gray = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2GRAY)
        model_gray = cv2.resize(model_gray, (orig_rgb.shape[1], orig_rgb.shape[0]))

        # Convert original → LAB
        orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB)
        l_orig, a_orig, b_orig = cv2.split(orig_lab)

        # Replace L channel with model output
        merged_lab = cv2.merge((model_gray, a_orig, b_orig))
        enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # ------------------- Post-processing -------------------
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        # CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        clahe_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Gamma correction
        gamma = 1.2
        look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                            for i in np.arange(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(clahe_rgb, look_up)

        # Sharpening
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        final = cv2.filter2D(gamma_corrected, -1, kernel)

        # Save result
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, final)

        print(f"✅ Saved {save_path} (proper enhanced RGB image)")

print("🎯 All images enhanced and saved in natural RGB!")
