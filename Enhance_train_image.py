import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"   # Low-light images
output_dir = "/content/outputs/train_enhanced"     # Save enhanced images
model_path = "/content/saved_model/cdan_denseunet.pt"  # Full .pt model file
os.makedirs(output_dir, exist_ok=True)
# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- Load model (.pt full model) --------
try:
    model = torch.load(model_path, map_location=device)
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}. Please check the path.")
    exit()
# -------- Preprocessing --------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   # Match your model input size
    transforms.ToTensor(),            # Converts to [0,1] tensor
])
# -------- Inference --------
with torch.no_grad():
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            inp = preprocess(img).unsqueeze(0).to(device)
            # Forward pass
            out = model(inp)
            # Clamp to [0,1] and remove batch dimension
            out = torch.clamp(out.squeeze(0), 0, 1).cpu()
            # Convert to PIL image
            out_img = to_pil_image(out)
            # Save enhanced image
            save_path = os.path.join(output_dir, fname)
            out_img.save(save_path)
            print(f"✅ Enhanced & saved: {fname}")
print("🎉 All images enhanced successfully!")
