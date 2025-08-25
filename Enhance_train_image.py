import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image
import sys
# -------- Add model path --------
sys.path.append('/content/CdanDenseUNet')  # adjust as needed
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light images
output_dir = "/content/outputs/train_enhanced"     # Where enhanced images will be saved
model_path = "/content/saved_model/cdan_denseunet.pt"  # Model weights
os.makedirs(output_dir, exist_ok=True)
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model --------
try:
    # Initialize the model
    model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
    # Load the weights (state dict)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Set to evaluation mode
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}. Please check the path.")
    sys.exit()
# -------- Preprocessing --------
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_tensor = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    return img_tensor
# -------- Run inference --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            inp = preprocess_image(img_path).to(device)
            # Forward pass
            out = model(inp)
            # Clamp to [0,1] and remove batch
            out = torch.clamp(out.squeeze(0), 0, 1).cpu()
            # If model outputs 1-channel, repeat to 3 channels
            if out.shape[0] == 1:
                out = out.repeat(3, 1, 1)
            # Convert to PIL and save
            out_img = to_pil_image(out)
            save_path = os.path.join(output_dir, fname)
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {fname}")
print("✅ All images processed successfully.")
