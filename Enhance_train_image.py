import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

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
    transforms.Resize((224, 224)),  # match training
    transforms.ToTensor()
])
to_pil = ToPILImage()

# ------------------- Inference + Enhancement Loop -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load original image
        img_path = os.path.join(input_dir, fname)
        orig_img = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(orig_rgb)

        inp = transform(pil_img).unsqueeze(0).to(device)

        # Forward pass through model
        out = model(inp).squeeze(0)  # [3,H,W]

        # --------- Per-channel normalization (preserves colors) ---------
        for c in range(3):
            cmin, cmax = out[c].min(), out[c].max()
            out[c] = (out[c] - cmin) / (cmax - cmin + 1e-8)

        out = torch.clamp(out, 0, 1)

        # Convert tensor -> numpy -> uint8
        out_np = out.permute(1, 2, 0).cpu().numpy()
        out_uint8 = (out_np * 255).astype(np.uint8)

        # ------------------- LAB Merge Strategy -------------------
        # Model output → replace only L channel
        out_lab = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2LAB)
        orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB)

        l_model, _, _ = cv2.split(out_lab)
        _, a_orig, b_orig = cv2.split(orig_lab)

        merged_lab = cv2.merge((l_model, a_orig, b_orig))
        enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # ------------------- Post-processing (optional) -------------------
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Save final enhanced image
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, final)

        # Debug info
        r_mean, g_mean, b_mean = final[..., 2].mean(), final[..., 1].mean(), final[..., 0].mean()
        print(f"{fname} -> R:{r_mean:.3f} G:{g_mean:.3f} B:{b_mean:.3f} | Saved ✅")

print("✅ All images enhanced with preserved colors and saved successfully!")
