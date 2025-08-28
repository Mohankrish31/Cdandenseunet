import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
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
model = CDANDenseUNet(in_channels=3, base_channels=32, output_range="01").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# ------------------- Preprocessing -------------------
# Use the same normalization as training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0,1] float tensor in CHW
])

# ------------------- Postprocessing -------------------
def postprocess_tensor(tensor):
    """
    Convert model output (0-1 float) to uint8 RGB image.
    Applies gamma correction (linear to sRGB).
    """
    # Clamp output
    tensor = torch.clamp(tensor, 0, 1)
    
    # Linear to sRGB
    tensor = torch.where(tensor <= 0.0031308,
                         tensor * 12.92,
                         1.055 * (tensor ** (1/2.4)) - 0.055)
    
    tensor = torch.clamp(tensor * 255.0, 0, 255).to(torch.uint8)
    return tensor

# ------------------- Inference -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)  # Add batch dim
        
        # Run model
        out = model(inp).squeeze(0).cpu()  # CHW, 0-1 float
        
        # Postprocess
        enhanced_img = postprocess_tensor(out)
        out_pil = to_pil_image(enhanced_img)
        
        # Save
        save_path = os.path.join(output_dir, fname)
        out_pil.save(save_path)
        
        print(f"{fname} -> min: {out.min().item():.4f}, max: {out.max().item():.4f}, mean per channel: {out.mean(dim=(1,2))}")
        print(f"✅ Enhanced & saved: {fname}")

print("\n✅ All images processed successfully!")
