import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light test images
output_dir = "/content/drive/MyDrive/Colon_Enhanced/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model and weights --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            # Model inference (raw output, no clamp yet)
            outs = model(inp).cpu().detach()
            # Debug: check raw output range
            print(f"🔎 {fname} -> raw output min: {outs.min().item():.4f}, max: {outs.max().item():.4f}")
            # Rescale output to [0,1] range
            outs = (outs - outs.min()) / (outs.max() - outs.min() + 1e-8)
            # Debug: check normalized range
            print(f"✅ {fname} -> normalized output min: {outs.min().item():.4f}, max: {outs.max().item():.4f}")
            # Process each output in batch
            for i in range(outs.size(0)):
                out_img = to_pil_image(outs[i])    # Tensor → PIL
                out_cv = np.array(out_img)         # PIL → NumPy
                # Save final enhanced image
                save_path = os.path.join(output_dir, f"enhanced_{fname}")
                Image.fromarray(out_cv).save(save_path)
                print(f"💾 Enhanced & saved (train): {save_path}")
print("🎉 All train images processed and saved to:", output_dir)
