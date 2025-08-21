import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"❌ Could not open image {fname}, skipping.")
            continue
        # Preprocess
        inp = transform(img).unsqueeze(0).to(device)
        # Forward pass
        out = model(inp)
        # Remove batch dimension
        out = out.squeeze(0).cpu()
        # Check for NaNs/Infs
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"⚠️ Output contains NaN/Inf for {fname}, skipping.")
            continue
        # Correct Rescaling Logic
        min_val = out.min()
        max_val = out.max()
        # Avoid division by zero
        if (max_val - min_val) > 1e-6:
            out = (out - min_val) / (max_val - min_val)
        else:
            print(f"⚠️ Output has no variance for {fname}, setting to black image.")
            out = torch.zeros_like(out)
        # Clamp to ensure values are within [0, 1]
        out = out.clamp(0, 1)
        # Convert to PIL
        out_img = to_pil_image(out)
        # Save enhanced image
        save_path = os.path.join(output_dir, f"enhanced_{fname}")
        out_img.save(save_path)
        print(f"✅ Enhanced & saved: {save_path}")
print("🎉 All training images processed and saved to:", output_dir)
