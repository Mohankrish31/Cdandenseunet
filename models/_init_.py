# int.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from cdan_denseunet import CDANDenseUNet   # import your model
def preprocess_image(image_path, resize_size=(224,224)):
    """Load and preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not read image: {image_path}")
    img = cv2.resize(img, resize_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR->RGB
    img = img.astype(np.float32) / 255.0        # normalize to [0,1]
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
    return img_tensor
def save_output(tensor, save_path):
    """Convert tensor to image and save"""
    out_img = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    out_img = (out_img * 255.0).clip(0,255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)  # back to BGR for cv2
    cv2.imwrite(save_path, out_img)
    print(f"✅ Saved enhanced image at {save_path}")
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------- 1. Initialize model -------------------
    model = CDANDenseUNet(
        in_ch=3,
        out_ch=3,
        growth_rate=12,
        block_layers=(3,4,5),
        base_channels=32,
        use_cbam_encoder=True,
        use_cbam_decoder=True
    ).to(device)
    # ------------------- 2. Load trained weights -------------------
    weight_path = "cdan_denseunet.pth"  # change this to your trained weights
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f"💾 Loaded trained weights from {weight_path}")
    # ------------------- 3. Load and preprocess input image -------------------
    input_path = "test.jpg"        # change this to your input image
    output_path = "enhanced.jpg"   # output path
    img_tensor = preprocess_image(input_path).to(device)
    # ------------------- 4. Forward pass -------------------
    with torch.no_grad():
        gen_out, feat_vec = model(img_tensor)
    # ------------------- 5. Save enhanced output -------------------
    save_output(gen_out, output_path)
if __name__ == "__main__":
    main()
