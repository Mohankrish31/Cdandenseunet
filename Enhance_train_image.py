import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.cdan_cbam_densenet import CDAN_CBAM_DenseNet  # Make sure your model code is in models/cdan_cbam_densenet.py

# ========================= SETTINGS =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/content/saved_model/cdan_cbam_densenet.pth"        # Path to your trained checkpoint
input_dir = "/content/cvccolondbsplit/train/low"           # Folder with images to enhance
output_dir = "/content/outputs/train_enhanced"            # Folder to save enhanced outputs
os.makedirs(output_dir, exist_ok=True)

# ========================= LOAD MODEL =========================
model = CDAN_CBAM_DenseNet().to(device)

if os.path.exists(model_path):
    # Load pretrained weights
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Loaded pretrained weights")
else:
    # No checkpoint found, save the randomly initialized model for future use
    torch.save(model.state_dict(), model_path)
    print(f"⚠️ No checkpoint found. Saved random weights to {model_path}")

model.eval()

# ========================= TRANSFORMS =========================
resize_dim = (224, 224)  # Optional: comment this out to keep original size
transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.ToTensor()  # Converts to [0,1]
])

# ========================= BATCH INFERENCE =========================
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load and preprocess image
        img_path = os.path.join(input_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]

        # Forward pass
        output_tensor = model(input_tensor).squeeze(0).cpu().clamp(0,1)  # [3,H,W]

        # Save output
        output_image = transforms.ToPILImage()(output_tensor)
        save_path = os.path.join(output_dir, fname)
        output_image.save(save_path)
        print(f"✅ Saved: {save_path}")

        # Optional: Display original vs enhanced
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(pil_img)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title("Enhanced")
        plt.imshow(output_image)
        plt.axis('off')
        plt.show()

print("🎯 All images processed successfully!")
