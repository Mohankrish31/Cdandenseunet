import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import sys
# -------- Add model path --------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light images
output_dir = "/content/outputs/train_enhanced"     # Enhanced images save path
model_path = "/content/saved_model/cdan_denseunet.pt"  # Model file
os.makedirs(output_dir, exist_ok=True)
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model --------
try:
    model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}. Please check the path.")
    sys.exit()
# -------- Preprocessing --------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Normalizes to [0,1]
])
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
            # Clamp to [0,1] and remove batch
            out = torch.clamp(out.squeeze(0), 0, 1).cpu()
            # If model outputs grayscale (1-channel), expand to 3 channels
            if out.shape[0] == 1:
                out = out.repeat(3, 1, 1)
            # Convert to PIL and save
            out_img = to_pil_image(out)
            save_path = os.path.join(output_dir, fname)
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {fname}")
