import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from PIL import Image
def simulate_endoscopic_degradation(image, brightness_range=(0.85, 0.95), gaussian_blur_sigma=0.8, noise_level=0.005):
    """
    Realistic yet CONTROLLED low-light simulation for colonoscopy images.
    Parameters are tuned to avoid black outputs while maintaining realism.
    Simplified by removing artificial motion blur kernel.
    """
    height, width, _ = image.shape
    degraded_image = image.astype(np.float32) # Convert to float for operations

    # 1. MUCH MILDER Vignetting (The MAIN CULPRIT)
    center_x, center_y = width // 2, height // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    normalized_dist = dist_from_center / max_dist
    
    # Use a much smaller exponent for a gentler falloff
    exponent = random.uniform(0.5, 0.8)
    vignette_mask = 1 - (normalized_dist**exponent * 0.4)  # Only reduce corners by max 40%
    vignette_mask = np.stack([vignette_mask]*3, axis=-1)
    
    # Apply the gentle vignette
    degraded_image *= vignette_mask

    # 2. Brightness Scaling - Apply LESS reduction
    brightness_factor = random.uniform(*brightness_range)
    degraded_image *= brightness_factor

    # 3. BLUR: ONLY Gaussian Blur for softness and mild motion effect
    # Removed the artificial motion blur kernel. Gaussian blur is more natural.
    if gaussian_blur_sigma > 0:
        kernel_size = 3  # Keep kernel small to avoid excessive blurring
        # Ensure kernel size is odd
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        degraded_image = cv2.GaussianBlur(degraded_image, (kernel_size, kernel_size), sigmaX=gaussian_blur_sigma)

    # 4. SMALL Amount of Noise
    noise = np.random.poisson(degraded_image * noise_level).astype(np.float32)
    degraded_image += noise

    # 5. Slight Color Cast
    red_scale = random.uniform(1.03, 1.10)
    green_scale = random.uniform(0.97, 1.03)
    blue_scale = random.uniform(0.94, 0.98)

    degraded_image[..., 0] *= blue_scale   # Blue channel
    degraded_image[..., 1] *= green_scale  # Green channel
    degraded_image[..., 2] *= red_scale    # Red channel

    # Clip and convert back to uint8
    degraded_image = np.clip(degraded_image, 0, 255).astype(np.uint8)
    return degraded_image
def prepare_dataset(input_dir, output_dir, val_ratio=0.1, test_ratio=0.2, resize_size=(224, 224)):
    """
    Prepares a low-light dataset by splitting a directory of high-res images
    and applying a custom degradation function. Saves images in RGB format.
    """
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_list.sort()
    if len(image_list) == 0:
        print("No images found in", input_dir)
        return
    trainval_imgs, test_imgs = train_test_split(image_list, test_size=test_ratio, random_state=42)
    train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=val_ratio / (1 - test_ratio), random_state=42)
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
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
                print(f"Skipping unreadable image: {fname}")
                continue
            image = cv2.resize(image, resize_size)
            degraded_img = simulate_endoscopic_degradation(image)
            # ✅ Save in RGB format instead of BGR
            rgb_high = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_low = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_high).save(os.path.join(high_dir, fname))
            Image.fromarray(rgb_low).save(os.path.join(low_dir, fname))
    print("\n✅ Dataset preparation complete. (Images saved in RGB format)")
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
