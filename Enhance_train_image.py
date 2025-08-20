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
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light train images
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
    transforms.ToTensor()
])
# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            # Model inference
            outs = model(inp).cpu().clamp(0, 1)  # [B, 3, H, W]
            # Process each output in batch
            for i in range(outs.size(0)):
                out_img = to_pil_image(outs[i])              # Tensor → PIL
                out_cv = np.array(out_img)                   # PIL → NumPy
                # Alternative (direct tensor → NumPy, HWC format):
                # out_cv = outs[i].cpu().detach().permute(1, 2, 0).numpy()
                # Save final enhanced image
                final_img = Image.fromarray(out_cv)
                save_path = os.path.join(output_dir, f"enhanced_{fname}")
                final_img.save(save_path)
                print(f"✅ Enhanced & saved (train): {save_path}")
print("🎉 All train images processed and saved to:", output_dir)
