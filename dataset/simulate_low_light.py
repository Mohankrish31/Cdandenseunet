import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')  # change if needed
from models.cdan_denseunet import CDANDenseUNet  # replace with your model class

# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/test/low"   # low-light images
output_dir = "/content/outputs/test_enhanced"   # save enhanced images
model_path = "/content/cdan_denseunet_weights.pth"  # trained model weights

os.makedirs(output_dir, exist_ok=True)

# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Load Model -------------------
model = CDANDenseUNet(in_channels=3, out_channels=3)  # adjust if needed
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------- Preprocessing -------------------
preprocess = transforms.Compose([
    transforms.ToTensor(),   # Converts HxWxC [0,255] to CxHxW [0,1]
])

# ------------------- Inference Loop -------------------
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Load low-light image
    img_path = os.path.join(input_dir, fname)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping unreadable image: {fname}")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # 1xCxHxW

    # Model inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocess and save
    output_image = output_tensor.squeeze(0).cpu().numpy()
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    output_image = np.transpose(output_image, (1, 2, 0))  # CxHxW -> HxWxC
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(os.path.join(output_dir, fname), output_image)
    print(f"✅ Enhanced & saved: {fname}")

print("\nAll images processed successfully.")
