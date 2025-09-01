import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage, Resize
# ------------------- Add model folder -------------------
# NOTE: Ensure this path is correct for your environment
sys.path.append('/content/CdanDenseUNet') 
from models.cdan_denseunet import CDANDenseUNet

# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"     # folder with low-light input images
output_dir = "/content/outputs/train_enhanced"     # folder to save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pth"     # trained weights
os.makedirs(output_dir, exist_ok=True)

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDANDenseUNet(
    in_channels=3,
    out_channels=3,
    base_channels=24,      # MUST match your trained model
    growth_rate=12,
    output_range="01"      # outputs in [0,1]
)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model state dict: {e}")
    sys.exit(1) # Exit the script if the model can't be loaded

model.to(device)
model.eval()

# ------------------- Transforms -------------------
# Assuming your model was trained on 224x224 images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
to_pil = ToPILImage()  # converts tensor [0-1] -> PIL [0-255]

# ------------------- Inference Loop -------------------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        # Load and preprocess
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)  # shape will be [1, 3, 224, 224]
        
        # Forward pass
        out = model(inp)
        
        # Clamp output to [0,1] and post-process
        out = torch.clamp(out.squeeze(0), 0, 1)
        
        # Convert to PIL and save
        enhanced_img = to_pil(out.cpu())
        enhanced_img.save(os.path.join(output_dir, fname))
        
        # Debug info
        avg_rgb = out.mean().item()
        print(f"{fname} -> Avg RGB (0–1): {avg_rgb:.4f} | Saved ✅")

