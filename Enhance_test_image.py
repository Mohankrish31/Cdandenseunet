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
input_dir = "/content/cvccolondbsplit/test/low"   # Low-light test images
output_dir = "/content/drive/MyDrive/Colon_Enhanced/test_enhanced"
cdan_model_path = "/content/Cdandenseunet/saved_models/cdan_denseunet.pth"  # Make sure this is the correct weights file

# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)

# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Initialize model --------
cdan_model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)

# -------- Load weights safely --------
if not os.path.exists(cdan_model_path):
    raise FileNotFoundError(f"Model file not found at {cdan_model_path}. Upload the correct .pth/.pt file first.")

try:
    checkpoint = torch.load(cdan_model_path, map_location=device)
    # Check if it is a full checkpoint or just state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        cdan_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        cdan_model.load_state_dict(checkpoint, strict=False)
    cdan_model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model. Ensure {cdan_model_path} is a valid PyTorch weights file. Error: {e}")

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# -------- Enhance and save testing images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            
            # Model inference
            out = cdan_model(inp).squeeze().cpu().clamp(0, 1)
            out_img = to_pil(out)
            
            # Save result
            final_img = Image.fromarray(np.array(out_img))
            final_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved: {fname}")

print("🎉 All test images processed and saved to:", output_dir)
