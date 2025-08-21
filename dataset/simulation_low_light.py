import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
# === Custom Dataset Class with Improved Low-Light Simulation ===
class cvccolondbsplitDataset(Dataset):
    def __init__(self, low_root, high_root, transform_low=None, simulate=False):
        self.low_root = low_root
        self.high_root = high_root
        self.transform_low = transform_low
        self.simulate = simulate  # Apply low-light simulation if True
        self.to_tensor = transforms.ToTensor()
        self.filenames = sorted(os.listdir(high_root))  # Assuming same filenames in both folders
    def __len__(self):
        return len(self.filenames)
    def simulate_low_light(self, image):
        """
        Improved low-light simulation:
        1. Moderate gamma correction to darken image.
        2. Add Poisson noise without subtraction to avoid negatives.
        3. Add Gaussian noise.
        4. Apply ColorJitter with higher brightness for very dark inputs.
        """
        # Convert PIL to NumPy, normalize to [0, 1]
        np_img = np.array(image).astype(np.float32) / 255.0
        # --- 1. Moderate Gamma Correction ---
        gamma = random.uniform(1.5, 2.5)  # Avoid extreme darkening
        low_light_img = np.power(np_img, gamma)
        # --- 2. Add Poisson Noise (signal-dependent noise) ---
        poisson_scaled = low_light_img * random.uniform(20, 50)
        poisson_noise = np.random.poisson(poisson_scaled) / random.uniform(20, 50)
        noisy_img = low_light_img + poisson_noise
        noisy_img = np.clip(noisy_img, 0, 1)
        # --- 3. Add Gaussian Noise (electronic noise) ---
        mean = 0
        std = random.uniform(0.01, 0.05)
        gaussian_noise = np.random.normal(mean, std, noisy_img.shape)
        noisy_img = noisy_img + gaussian_noise
        noisy_img = np.clip(noisy_img, 0, 1)
        # --- 4. Color Jitter (after noise) ---
        jitter = transforms.ColorJitter(
            brightness=random.uniform(0.1, 0.3),  # Higher brightness for dark images
            contrast=random.uniform(0.0, 0.1),
            saturation=random.uniform(0.0, 0.1),
            hue=random.uniform(-0.05, 0.05)
        )
        pil_img = Image.fromarray((noisy_img * 255).astype(np.uint8))
        jittered_img = jitter(pil_img)
        # Convert back to NumPy and clamp
        final_img = np.array(jittered_img).astype(np.float32) / 255.0
        final_img = np.clip(final_img, 0, 1)
        return Image.fromarray((final_img * 255).astype(np.uint8))
    def __getitem__(self, idx):
        high_img_path = os.path.join(self.high_root, self.filenames[idx])
        high = Image.open(high_img_path).convert('RGB')
        if self.simulate:
            low = self.simulate_low_light(high.copy())
        else:
            low_img_path = os.path.join(self.low_root, self.filenames[idx])
            low = Image.open(low_img_path).convert('RGB')
        if self.transform_low:
            low = self.transform_low(low)
        else:
            low = self.to_tensor(low)
        high = self.to_tensor(high)
        return low, high
# === Transforms ===
# Strong Augmentation for Training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
# No Augmentation for Validation and Testing
val_test_transform = transforms.ToTensor()
# === Dataset Setup ===
train_dataset = cvccolondbsplitDataset(
    low_root='/content/cvccolondbsplit/train/low',
    high_root='/content/cvccolondbsplit/train/high',
    transform_low=train_transform,
    simulate=True  # Generate low-light from high
)
val_dataset = cvccolondbsplitDataset(
    low_root='/content/cvccolondbsplit/val/low',
    high_root='/content/cvccolondbsplit/val/high',
    transform_low=val_test_transform,
    simulate=False  # Use provided low-light
)
test_dataset = cvccolondbsplitDataset(
    low_root='/content/cvccolondbsplit/test/low',
    high_root='/content/cvccolondbsplit/test/high',
    transform_low=val_test_transform,
    simulate=False  # Use provided low-light
)
