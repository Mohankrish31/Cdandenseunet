# ==============================
# Requirements for CDAN-DenseUNet
# Low-light colonoscopy image enhancement
# ==============================
# ------------------------------
# PyTorch & CUDA
# Ensure torch version matches your CUDA version
# Example for CUDA 11.7: torch==1.13.1+cu117
# ------------------------------
torch>=1.10
torchvision>=0.11
# ------------------------------
# Perceptual Loss
# LPIPS: Learned Perceptual Image Patch Similarity
# ------------------------------
lpips>=0.1.4
# ------------------------------
# Basic Python packages
# ------------------------------
numpy>=1.19
Pillow>=8.0
tqdm>=4.60
matplotlib>=3.3
opencv-python
scikit-image

