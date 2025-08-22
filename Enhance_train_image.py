import os
import torch
from PIL import Image
from torchvision import transforms
import sys

# -------- Add model path --------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet

# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"  # Your low-light images
output_dir = "/content/outputs/train_enhanced"  # Where enhanced images will be saved
model_path = "/content/saved_model/cdan_denseunet.pt" # Correct path to the model file
os.makedirs(output_dir, exist_ok=True)

# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load model --------
try:
    model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
    # The original script had a small typo in the filename.
    # The correct filename appears to be 'cdan_denseunet.pt'.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}. Please check the path.")
    sys.exit()

# -------- Preprocessing and Post-processing --------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Normalizes to [0, 1] range
])
# No separate ToPILImage needed; we'll use a direct method
# to convert the tensor to a PIL image.

# -------- Enhance and save images --------
# -------- Run inference --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            inp = preprocess(img).unsqueeze(0).to(device)

            # Forward pass
            out = model(inp)

            # If model outputs [-1, 1], rescale -> [0, 1]
            out = (out + 1) / 2  

            # Clamp and convert
            out_clamped = out.squeeze(0).cpu().clamp(0, 1)
            out_img = transforms.ToPILImage()(out_clamped)

            save_path = os.path.join(output_dir, fname)
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {fname}")


