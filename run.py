import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
# -------- Add repo root to sys.path if needed --------
sys.path.append('/content/Cdandenseunet')  # Adjust path if needed in Colab
# -------- Import utils --------
from utils.dataloader import cvccolondbDataset
from utils.custom_loss import 
from utils.metrics import calculate_psnr, calculate_ssim, calculate_ebcm, lpips_fn
from utils.plot_loss import plot_loss_curve
from utils.plot_metrics import plot_metrics_curve
# -------- Import model --------
from models.cdan_denseunet import cdan_denseunet

# -------- Argument parser --------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], required=True,
                    help="Choose mode: train, val, or test")
args = parser.parse_args()

# -------- Set directories based on mode --------
if args.mode == "train":
    input_dir = "/content/cvccolondbsplit/train/low"
    high_dir = "/content/cvccolondbsplit/train/high"
    output_dir = "/content/drive/MyDrive/Colon_Enhanced/train_enhanced"
elif args.mode == "val":
    input_dir = "/content/cvccolondbsplit/val/low"
    high_dir = "/content/cvccolondbsplit/val/high"
    output_dir = "/content/drive/MyDrive/Colon_Enhanced/val_enhanced"
elif args.mode == "test":
    input_dir = "/content/cvccolondbsplit/test/low"
    high_dir = "/content/cvccolondbsplit/test/high"
    output_dir = "/content/drive/MyDrive/Colon_Enhanced/test_enhanced"
os.makedirs(output_dir, exist_ok=True)

# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load model --------
model_path = "/content/models/cdan_denseunet.pt"
model = cdan_denseunet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# -------- Enhance and save images --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            # Model inference
            out = model(inp).squeeze().cpu().clamp(0, 1)
            out_img = to_pil(out)
            final_img = Image.fromarray(np.array(out_img))
            final_img.save(os.path.join(output_dir, fname))
            print(f"✅ Enhanced & saved ({args.mode}): {fname}")

print(f"🎉 All {args.mode} images processed and saved to: {output_dir}")

# -------- Optional: Create Dataset & DataLoader for metrics / plotting --------
dataset = cvccolondbDataset(enhanced_dir=output_dir, high_dir=high_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# -------- Optional: Calculate metrics --------
psnr_list, ssim_list, ebcm_list, lpips_list = [], [], [], []

for enhanced_img, high_img in loader:
    enhanced_img = enhanced_img.to(device)
    high_img = high_img.to(device)
    
    psnr_val = calculate_psnr(enhanced_img, high_img)
    ssim_val = calculate_ssim(enhanced_img, high_img)
    ebcm_val = calculate_ebcm(enhanced_img, high_img)
    lpips_val = lpips_fn(enhanced_img, high_img)
    
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    ebcm_list.append(ebcm_val)
    lpips_list.append(lpips_val)

# -------- Optional: Plot metrics --------
plot_metrics(psnr_list, ssim_list, ebcm_list, lpips_list)
