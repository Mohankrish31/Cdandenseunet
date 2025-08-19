import os
import cv2
import torch
import lpips
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.cdan_denseunet import cdan_denseunet
from skimage.metrics import structural_similarity as compare_ssim
import math
# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define directories
input_dir = "/content/cvccolondbsplit/test/input"
output_dir = "/content/outputs/test_enhanced"
gt_dir = "/content/cvccolondbsplit/test/high"
os.makedirs(output_dir, exist_ok=True)
# === Initialize model ===
model = cdan_denseunet(in_channels=3, base_channels=32).to(device)
model.load_state_dict(torch.load("best_cdan_denseunet.pth"))
model.eval()
# === Initialize LPIPS ===
lpips_fn = lpips.LPIPS(net='vgg').to(device)
# === Metric Functions ===
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)
def calculate_ssim(img1, img2):
    ssim = 0
    for i in range(3):
        ssim += compare_ssim(img1[..., i], img2[..., i], data_range=1.0)
    return ssim / 3
def calculate_ebcm(img1, img2):
    gray1 = cv2.cvtColor((img1*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag1 = np.sqrt(sobelx1**2 + sobely1**2)
    edge_mag2 = np.sqrt(sobelx2**2 + sobely2**2)
    edge_mag1[edge_mag1==0] = 1e-6
    return np.mean(np.minimum(edge_mag1, edge_mag2)/np.maximum(edge_mag1, edge_mag2))
# === Run Inference and Evaluate ===
results = []
for fname in tqdm(sorted(os.listdir(input_dir)), desc="Enhancing Images"):
    input_path = os.path.join(input_dir, fname)
    img = cv2.imread(input_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
    input_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)
    # --- Inference ---
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
    enhanced_img = enhanced_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    enhanced_img = np.clip(enhanced_img, 0, 1)
    # --- Save enhanced image ---
    save_path = os.path.join(output_dir, fname)
    save_bgr = cv2.cvtColor((enhanced_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_bgr)
    # --- Compute metrics if ground truth exists ---
    # Optional: Replace 'gt_dir' with folder containing high-quality reference images
    gt_dir = "/content/cvccolondbsplit/test/high"
    gt_path = os.path.join(gt_dir, fname)
    if os.path.exists(gt_path):
        gt_img = cv2.imread(gt_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)/255.0
        if gt_img.shape != enhanced_img.shape:
            enhanced_img = cv2.resize(enhanced_img, (gt_img.shape[1], gt_img.shape[0]))
        psnr = calculate_psnr(enhanced_img, gt_img)
        ssim = calculate_ssim(enhanced_img, gt_img)
        ebcm = calculate_ebcm(enhanced_img, gt_img)
        with torch.no_grad():
            enhanced_tensor_lp = torch.tensor(enhanced_img).permute(2,0,1).unsqueeze(0).float().to(device)
            gt_tensor_lp = torch.tensor(gt_img).permute(2,0,1).unsqueeze(0).float().to(device)
            lpips_val = lpips_fn(enhanced_tensor_lp, gt_tensor_lp).item()
        results.append({
            "filename": fname,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_val,
            "EBCM": ebcm
        })
        print(f"{fname}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_val:.4f}, EBCM={ebcm:.4f}")
# === Save metrics to CSV if available ===
if results:
    df = pd.DataFrame(results)
    df.to_csv("metrics_results_test.csv", index=False)
    print("\n✅ Metrics saved to 'metrics_results_test.csv'")
