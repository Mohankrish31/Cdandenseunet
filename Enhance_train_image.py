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
input_dir = "/content/cvccolondbsplit/train/low"  # low-light images
output_dir = "/content/outputs/train_enhanced"  # save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pth"  # trained weights
os.makedirs(output_dir, exist_ok=True)

# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Post-processing (Gamma Correction) -------------------
def linear_to_srgb(image):
    """
    Converts a linear image (0-1 float) to an sRGB image (0-255 uint8).
    This is a crucial post-processing step.
    """
    srgb_image = torch.where(image <= 0.0031308,
                             image * 12.92,
                             1.055 * (image ** (1/2.4)) - 0.055)
    srgb_image = torch.clamp(srgb_image * 255.0, 0, 255).to(torch.uint8)
    return srgb_image

# ------------------- Load Model -------------------
model = CDANDenseUNet(in_channels=3, base_channels=32, output_range="01").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")

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

        # --- Preview input image with OpenCV ---
        img_cv = cv2.imread(img_path)
        cv2.imshow("Input Image", img_cv)
        cv2.waitKey(0)  # Press any key to continue
        cv2.destroyAllWindows()

        # --- Run preprocessing ---
        inp = preprocess_image(img_path).to(device)

        # Run the model
        out = model(inp).squeeze(0).cpu()
        out = torch.clamp(out, 0, 1)

        # Print output tensor stats
        print(f"{fname} -> min: {out.min().item()}, max: {out.max().item()}, mean per channel: {out.mean(dim=(1,2))}")

        # Post-processing
        enhanced_img = linear_to_srgb(out)

        # Convert to PIL Image and save
        out_pil = to_pil_image(enhanced_img)
        save_path = os.path.join(output_dir, fname)
        out_pil.save(save_path)
        print(f"✅ Enhanced & saved: {fname}")

print("\n✅ All images processed successfully!")
