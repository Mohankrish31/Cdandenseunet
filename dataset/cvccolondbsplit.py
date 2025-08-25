import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
def simulate_low_light(image, brightness_range=(0.5, 0.8),
                       gamma_range=(1.0, 1.5),
                       poisson_scale=(5, 20),
                       gaussian_std=(0.005, 0.02)):
    """
    Simulate realistic low-light image with:
    - brightness reduction
    - gamma correction
    - Poisson noise
    - Gaussian noise
    """
    # Convert to HSV for brightness adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(*brightness_range)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    # Convert back to BGR
    low_light = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Apply gamma correction
    gamma = random.uniform(*gamma_range)
    low_light = np.power(low_light / 255.0, gamma) * 255.0
    # Apply Poisson noise
    scale = random.uniform(*poisson_scale)
    poisson_noisy = np.random.poisson(low_light * scale) / scale
    low_light = np.clip(poisson_noisy, 0, 255)
    # Apply Gaussian noise
    mean = 0
    std = random.uniform(*gaussian_std) * 255
    gaussian_noise = np.random.normal(mean, std, low_light.shape)
    low_light = np.clip(low_light + gaussian_noise, 0, 255).astype(np.uint8)
    return low_light
def prepare_dataset(input_dir, output_dir, val_ratio=0.1, test_ratio=0.2, resize_size=(224, 224)):
    # Read all images
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_list.sort()
    if len(image_list) == 0:
        print("No images found in", input_dir)
        return
    # Train/Val/Test split
    trainval_imgs, test_imgs = train_test_split(image_list, test_size=test_ratio, random_state=42)
    train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=val_ratio / (1 - test_ratio), random_state=42)
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    # Process and save images
    for split, filenames in splits.items():
        print(f"\nProcessing {split.upper()} split with {len(filenames)} images...")
        low_dir = os.path.join(output_dir, split, 'low')
        high_dir = os.path.join(output_dir, split, 'high')
        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(high_dir, exist_ok=True)
        for fname in tqdm(filenames):
            image_path = os.path.join(input_dir, fname)
            image = cv2.imread(image_path)
            if image is None:
                print("Skipping unreadable image:", fname)
                continue
            # Resize
            image = cv2.resize(image, resize_size)
            # Generate low-light image
            low_light_img = simulate_low_light(image)
            # Save high and low images
            cv2.imwrite(os.path.join(high_dir, fname), image)
            cv2.imwrite(os.path.join(low_dir, fname), low_light_img)
    print("\n✅ Dataset preparation complete.")
# === Run script ===
if __name__ == "__main__":
    original_dataset_path = "/content/cvccolondb/data/train/images"
    output_dataset_path = "/content/cvccolondbsplit"
    prepare_dataset(
        input_dir=original_dataset_path,
        output_dir=output_dataset_path,
        val_ratio=0.1,
        test_ratio=0.2,
        resize_size=(224, 224)
    )
