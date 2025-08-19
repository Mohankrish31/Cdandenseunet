import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import sys
from models.cdan_denseunet import cdan_denseunet

# -------- Argument parser --------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train','val','test'], required=True,
                    help="Choose mode: train, val, or test")
args = parser.parse_args()

# -------- Set directories based on mode --------
if args.mode == "train":
    input_dir = "/content/cvccolondbsplit/train/low"
    output_dir = "/content/outputs/train_enhanced"
elif args.mode == "val":
    input_dir = "/content/cvccolondbsplit/val/low"
    output_dir = "/content/outputs/val_enhanced"
elif args.mode == "test":
    input_dir = "/content/cvccolondbsplit/test/low"
    output_dir = "/content/outputs/test_enhanced"

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
