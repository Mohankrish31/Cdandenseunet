CDAN-DenseUNet (WIP)

Low-light colonoscopy image enhancement using Convolutional Dense Attention Network (CDAN) with DenseUNet backbone.

Status: Work in Progress – Training, validation, and testing scripts are functional. Results will be updated after experiments.

🚀 Features

CDAN Attention: Enhances feature maps via channel-dependent attention.

DenseUNet Backbone: Dense connections for better feature reuse and gradient flow.

Multi-Loss Optimization: Combines MSE, SSIM, LPIPS, Edge, and Range loss for perceptual and structural quality.

📂 Dataset

Dataset: CVC-ColonDB

Training & Validation Resolution: 224×224

Testing Resolution: Original dimensions (e.g., 574×500)

Augmentations: Random crop, flip, rotation

Split: Train / Validation / Test

📊 Evaluation Metrics

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

LPIPS (Learned Perceptual Image Patch Similarity)

EBCM (Edge-Based Contrast Measure)

🛠 Installation
git clone https://github.com/<your-username>/cdan-denseunet.git
cd cdan-denseunet
pip install -r requirements.txt
