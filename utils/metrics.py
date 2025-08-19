import os
import cv2
import torch
import lpips
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import math
import pandas as pd
# === Initialize LPIPS === #
lpips_fn = lpips.LPIPS(net='vgg').cuda()
# === PSNR === #
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 10 * math.log10((PIXEL_MAX ** 2) / mse)
# === SSIM === #
def calculate_ssim(img1, img2):
    ssim = 0
    for i in range(3):
        ssim += compare_ssim(img1[..., i], img2[..., i], data_range=1.0)
    return ssim / 3
# === EBCM (Edge-Based Contrast Measure) === #
def calculate_ebcm(img1, img2):
    """
    EBCM measures how well edges are preserved between reference and enhanced images.
    Uses Sobel edge maps for computation.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Compute Sobel edges
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag1 = np.sqrt(sobelx1**2 + sobely1**2)
    edge_mag2 = np.sqrt(sobelx2**2 + sobely2**2)
    # Avoid division by zero
    edge_mag1[edge_mag1 == 0] = 1e-6
    # EBCM formula: ratio of edge magnitudes, averaged
    ebcm_score = np.mean(np.minimum(edge_mag1, edge_mag2) / np.maximum(edge_mag1, edge_mag2))
    return ebcm_score
  # === Evaluation Function === #
def evaluate_metrics_individual(high_dir, enhanced_dir):
    filenames = sorted(os.listdir(high_dir))
    psnr_list, ssim_list, lpips_list, ebcm_list = [], [], [], []
    results = []
    for fname in tqdm(filenames, desc="🔍 Evaluating"):
        high_path = os.path.join(high_dir, fname)
        enh_path = os.path.join(enhanced_dir, fname)
        high_img = cv2.imread(high_path)
        enh_img = cv2.imread(enh_path)
        if high_img is None or enh_img is None:
            continue
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB) / 255.0
        enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB) / 255.0
        # Resize if shapes mismatch
        if high_img.shape != enh_img.shape:
            enh_img = cv2.resize(enh_img, (high_img.shape[1], high_img.shape[0]))
        # --- Compute PSNR ---
        psnr = calculate_psnr(enh_img, high_img)
        # --- Compute SSIM ---
        ssim = calculate_ssim(enh_img, high_img)
        # --- Compute EBCM ---
        ebcm = calculate_ebcm(high_img, enh_img)
        # --- Compute LPIPS ---
        with torch.no_grad():
            high_tensor = torch.tensor(high_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            enh_tensor  = torch.tensor(enh_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            lpips_val = lpips_fn(enh_tensor, high_tensor).item()  # ✅ LPIPS computation
        # --- Append results ---
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val)
        ebcm_list.append(ebcm)
        print(f"{fname}:  PSNR: {psnr:.4f}  SSIM: {ssim:.4f}  LPIPS: {lpips_val:.4f}  EBCM: {ebcm:.4f}")
        results.append({
            "filename": fname,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_val,
            "EBCM": ebcm
        })
    # === Summary Metrics === #
    print("\n=== Summary Metrics ===")
    print(f"Mean PSNR: {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"Mean LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}")
    print(f"Mean EBCM: {np.mean(ebcm_list):.4f} ± {np.std(ebcm_list):.4f}")
    # === Save results to CSV === #
    df = pd.DataFrame(results)
    df.to_csv("metrics_results_with_ebcm.csv", index=False)
    print("\n✅ Saved results to 'metrics_results_with_ebcm.csv'.")
  # === Run the function === #
  high_dir = "/content/cvccolondbsplit/test/high"
 enhanced_dir = "/content/outputs/test_enhanced"
 evaluate_metrics_individual(high_dir, enhanced_dir)

