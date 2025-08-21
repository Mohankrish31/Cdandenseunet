import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

# -------- Add model path --------
sys.path.append('/content/CbamDenseUnet')
from models.cdan_denseunet import CDANDenseUNet

# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"

# -------- Create output dir --------
os.makedirs(output_dir, exist_ok=True)

# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model Architecture --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)

# -------- Load weights --------
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # If your model expects normalization, UNCOMMENT below:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

# -------- Enhance images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)

            out = model(inp)    # [1,3,224,224] or similar
            out = out.squeeze()
            # Check for correct shape
            if out.ndim == 3 and out.shape[0] == 3:
                out = out.cpu().clamp(0, 1)
                print("Output min/max:", out.min().item(), out.max().item())  # DEBUG: See tensor values
                out_img = to_pil(out)
                out_img.save(os.path.join(output_dir, fname))
                print(f"✅ Enhanced & saved (train): {fname}")
            else:
                print(f"Unexpected output shape for {fname}: {out.shape}")

print("🎉 All training images processed and saved to:", output_dir)
