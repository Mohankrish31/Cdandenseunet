import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# -------- Add model path --------
sys.path.append('/content/Cdandenseunet')
try:
    from models.cdan_denseunet import CDANDenseUNet
except ImportError:
    print("❌ Error: The model module 'cdan_denseunet' could not be imported.")
    print("Please check your model folder path.")
    sys.exit(1)
# -------- Paths --------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pt"
# -------- Create output directory --------
os.makedirs(output_dir, exist_ok=True)
# -------- Setup device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------- Load model and weights --------
print("Loading model and weights...")
try:
    model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model weights: {e}")
    print("Please check if the model file path is correct and the file is not corrupted.")
    sys.exit(1)
# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Values will be in [0,1]
])
# -------- Enhance and save images --------
print("Starting image enhancement process...")
with torch.no_grad():
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No image files found in the input directory.")
        sys.exit(1)
    for fname in image_files:
        img_path = os.path.join(input_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"❌ Could not open image {fname}, skipping.")
            continue
        inp = transform(img).unsqueeze(0).to(device)
        out = model(inp).squeeze(0).cpu()
        # --- SOLUTION FOR THE GREEN/PINK BOX ISSUE ---
        # The core problem is often invalid tensor values from the model.
        # 1. Check for NaN/Inf values.
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"⚠️ Output contains NaN/Inf for {fname}. This indicates a model failure.")
            print("Skipping to avoid a corrupted image.")
            continue
        # 2. Rescale the values dynamically to the [0, 1] range.
        # This handles models that output values outside the standard range.
        min_val = out.min()
        max_val = out.max()
        # Avoid division by zero, which would cause NaNs
        if (max_val - min_val) > 1e-6:
            out = (out - min_val) / (max_val - min_val)
        else:
            # If all values are the same, the model failed to learn;
            # create a black image as a clear output.
            print(f"⚠️ Output has no variance for {fname}, creating a black image.")
            out = torch.zeros_like(out)
        # 3. Clamp final values to ensure they are strictly between 0 and 1.
        out = out.clamp(0, 1)
        # Convert the tensor to a PIL Image
        out_img = to_pil_image(out)
        # Save the enhanced image
        save_path = os.path.join(output_dir, f"enhanced_{fname}")
        out_img.save(save_path)
        print(f"✅ Enhanced & saved: {save_path}")
print("🎉 All images processed and saved to:", output_dir)
