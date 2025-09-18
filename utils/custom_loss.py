import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lpips
import numpy as np
from models.cdan_denseunet import CDANDenseUNet
from utils import plot_loss_curve

# ---------- Hyperparams ----------
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 100
batch_size = 8
early_stopping_patience = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset (clear names) ----------
class CVCCDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('.png','.jpg','.jpeg')) and os.path.exists(os.path.join(high_dir, f))])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.image_names[idx])
        high_path = os.path.join(self.high_dir, self.image_names[idx])
        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")
        # NOTE: apply transforms deterministically (here we only use Resize+ToTensor)
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        return low_img, high_img

# ---------- Loss utilities (use your existing functions) ----------
mse_loss_fn = nn.MSELoss()
# edge_loss_fn, ssim_loss_fn, total_loss_fn assumed identical to yours (reuse)
# lpips model:
lpips_model = lpips.LPIPS(net='vgg').to(device)

# ---------- Paths & transforms (fix paths!) ----------
train_low_dir  = "/content/cvccolondbsplit/train/low"   # DEGRADE inputs
train_high_dir = "/content/cvccolondbsplit/train/high"  # GT
val_low_dir    = "/content/cvccolondbsplit/val/low"
val_high_dir   = "/content/cvccolondbsplit/val/high"

# Use only deterministic transforms for paired supervision (no ColorJitter here).
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = CVCCDataset(train_low_dir, train_high_dir, transform)
val_dataset   = CVCCDataset(val_low_dir, val_high_dir, transform)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Model and optimizer ----------
model = CDANDenseUNet(in_channels=3, out_channels=3, base_channels=32, growth_rate=12, output_range="01").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Loss weights (reuse your values)
w_mse   = 0.40
w_lpips = 0.10
w_edge  = 0.15
w_ssim  = 0.35

# Helper: compute total loss using model outputs
def compute_losses(pred, target):
    # pred, target in [0,1]
    mse   = mse_loss_fn(pred, target)
    edge  = edge_loss_fn(pred, target)              # using your SobelEdgeLoss
    lp    = lpips_model(2*pred - 1, 2*target - 1).mean()  # lpips expects [-1,1]
    ssim  = ssim_loss_fn(pred, target)               # your SSIMLoss
    total = w_mse*mse + w_lpips*lp + w_edge*edge + w_ssim*ssim
    return total, mse, lp, edge, ssim

# ---------- Training loop (fixed: forward pass included) ----------
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for input_img, target_img in train_loader:
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()
        pred = model(input_img)                       # <-- IMPORTANT: forward pass
        total_loss, mse_val, lp_val, edge_val, ssim_val = compute_losses(pred, target_img)
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for input_img, target_img in val_loader:
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            pred = model(input_img)
            total_loss, _, _, _, _ = compute_losses(pred, target_img)
            val_running += total_loss.item()

    avg_val_loss = val_running / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}  TrainLoss: {avg_train_loss:.6f}  ValLoss: {avg_val_loss:.6f}")

    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cdan_denseunet.pth')
        print("Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping")
            break

# Plot
plot_loss_curve(train_losses, val_losses)
