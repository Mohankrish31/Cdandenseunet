import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
def srgb_to_linear(image):
    """
    Converts an sRGB image (0-255 uint8) to a linear image (0-1 float).
    This is a critical first step to simulate a camera's sensor.
    """
    image = image.astype(np.float32) / 255.0
    linear_image = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    return linear_image
def linear_to_srgb(image):
    """
    Converts a linear image (0-1 float) to an sRGB image (0-255 uint8).
    This is the final step to prepare the image for display.
    """
    srgb_image = np.where(image <= 0.0031308, image * 12.92, 1.055 * (image ** (1/2.4)) - 0.055)
    srgb_image = np.clip(srgb_image * 255.0, 0, 255).astype(np.uint8)
    return srgb_image
def simulate_low_light_isp(image, exposure_range=(0.02, 0.1),
                             white_balance_range=(0.8, 1.2),
                             poisson_gain_range=(0.02, 0.1),
                             read_noise_std=(0.001, 0.005),
                             brightness_scaling_factor=(0.1, 0.5)):
    """
    Simulates low-light conditions by following a corrected ISP pipeline.
    """
    # 1. Convert to Linear Space (Inverse Gamma) ⏪
    linear_img = srgb_to_linear(image)
    # 2. Simulate Low Exposure ⏪
    exposure = random.uniform(*exposure_range)
    low_exposure_img = linear_img * exposure
    # 3. Apply Combined Poisson-Gaussian Noise ⏪
    poisson_gain = random.uniform(*poisson_gain_range)
    poisson_noise = np.random.poisson(low_exposure_img / poisson_gain) * poisson_gain
    read_noise_stdev = random.uniform(*read_noise_std)
    gaussian_noise = np.random.normal(0, read_noise_stdev, linear_img.shape)
    noisy_linear_img = poisson_noise + gaussian_noise
    # 4. Apply White Balance ⏪
    white_balance_r = random.uniform(*white_balance_range)
    white_balance_g = random.uniform(*white_balance_range)
    white_balance_b = random.uniform(*white_balance_range)
    # Apply gains to noisy linear image
    noisy_linear_img[:, :, 2] *= white_balance_b  # B
    noisy_linear_img[:, :, 1] *= white_balance_g  # G
    noisy_linear_img[:, :, 0] *= white_balance_r  # R
    # 5. Apply Global Brightness Scaling ⏪
    # This step is key to avoiding the "gray box" issue
    brightness_factor = random.uniform(*brightness_scaling_factor)
    noisy_linear_img = noisy_linear_img * brightness_factor
    # 6. Clip and Convert to sRGB Space (Gamma Correction) ⏪
    low_light_srgb = linear_to_srgb(np.clip(noisy_linear_img, 0, 1))
    return low_light_srgb
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
            # Generate low-light image with the new ISP-based function
            low_light_img = simulate_low_light_isp(image)
            # Save high and low images
            cv2.imwrite(os.path.join(high_dir, fname), image)
            cv2.imwrite(os.path.join(low_dir, fname), low_light_img)  
    print("\n✅ Dataset preparation complete.")
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
