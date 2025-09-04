import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# ------------------- Add model folder -------------------
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet

# ------------------- Paths -------------------
input_dir = "/content/cvccolondbsplit/train/low"
output_dir = "/content/outputs/train_enhanced"
model_path = "/content/saved_model/cdan_denseunet.pth"
os.makedirs(output_dir, exist_ok=True)

# ------------------- Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- Load Model -------------------
print("Loading model...")
model = CDANDenseUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,
    growth_rate=12,
    output_range="01"
)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

model.to(device)
model.eval()
print("Model moved to device and set to evaluation mode")

# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match training
    transforms.ToTensor()
])

# ------------------- Image Enhancement Functions -------------------
def apply_clahe(image):
    """Apply CLAHE contrast enhancement"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_mild_sharpening(image):
    """Apply mild sharpening filter"""
    kernel = np.array([[0, -0.25, 0],
                       [-0.25, 2, -0.25],
                       [0, -0.25, 0]])
    return cv2.filter2D(image, -1, kernel)

def auto_white_balance(image):
    """Simple auto white balance"""
    result = image.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def normalize_image(image):
    """Normalize image to 0-255 range"""
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
    return image.astype(np.uint8)

def post_process_output(output_tensor):
    """Process model output to get normal RGB image"""
    # Convert tensor to numpy
    output_np = output_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert to 0-255 range
    output_uint8 = (output_np * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)
    
    return output_bgr

# ------------------- Inference Loop -------------------
print(f"Starting inference on {len([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])} images...")
print("-" * 60)

total_time = 0
processed_count = 0

with torch.no_grad():
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        start_time = time.time()
        
        try:
            # Load and preprocess image
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            original_size = img.size  # Store original size
            
            # Transform for model input
            inp = transform(img).unsqueeze(0).to(device)

            # Forward pass
            out = model(inp).squeeze(0)  # [3, H, W]
            out = torch.clamp(out, 0, 1)  # Ensure valid range

            # Post-process output
            enhanced_bgr = post_process_output(out)
            
            # Resize back to original dimensions
            enhanced_bgr = cv2.resize(enhanced_bgr, original_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply final enhancements
            enhanced_bgr = apply_clahe(enhanced_bgr)
            enhanced_bgr = apply_mild_sharpening(enhanced_bgr)
            enhanced_bgr = auto_white_balance(enhanced_bgr)
            
            # Save image
            output_path = os.path.join(output_dir, fname)
            cv2.imwrite(output_path, enhanced_bgr)
            
            # Calculate processing time
            end_time = time.time()
            process_time = end_time - start_time
            total_time += process_time
            processed_count += 1
            
            # Get color stats for debugging
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            r_mean, g_mean, b_mean = enhanced_rgb[...,0].mean(), enhanced_rgb[...,1].mean(), enhanced_rgb[...,2].mean()
            
            print(f"✅ {fname:20s} | Time: {process_time:.3f}s | RGB: ({r_mean:.1f}, {g_mean:.1f}, {b_mean:.1f})")
            
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            continue

# ------------------- Summary -------------------
if processed_count > 0:
    avg_time = total_time / processed_count
    print("-" * 60)
    print(f"🎉 Processing completed!")
    print(f"📊 Total images processed: {processed_count}")
    print(f"⏱️  Average time per image: {avg_time:.3f}s")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"💾 Output directory: {output_dir}")
else:
    print("❌ No images were processed. Check input directory and file formats.")

print("✅ Inference script completed!")
