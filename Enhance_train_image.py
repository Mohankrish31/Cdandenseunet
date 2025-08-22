import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
import sys
# -------- Add model path --------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Your low-light images
output_dir = "/content/outputs/train_enhanced"    # Where enhanced images will be saved
model_path = "/content/saved_model/cdandenseunet.pt"
os.makedirs(output_dir, exist_ok=True)
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model --------
model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load("/content/saved_model/cdan_denseunet.pt", map_location=device))
model.eval()
# -------- Preprocessing (no normalization) --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # raw [0,1] input
])
to_pil = ToPILImage()
# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
            
            out = model(inp)  # [1,3,H,W]

            # Remove batch dimension and ensure tensor is contiguous
            out = out.squeeze(0).cpu().clamp(0,1)

            # If needed, auto scale each channel
            min_val = out.min()
            max_val = out.max()
            if min_val < 0 or max_val > 1:
                out = (out - min_val) / (max_val - min_val + 1e-8)
                out = out.clamp(0, 1)

            # Convert to PIL image (make sure to use float32 in [0,1])
            out_img = to_pil(out.float())
            out_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved: {fname} | range: {min_val:.3f}-{max_val:.3f}")
