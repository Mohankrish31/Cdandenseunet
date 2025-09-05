import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from PIL import Image
def simulate_endoscopic_degradation(image, brightness_range=(0.7, 0.9), blur_range=(1, 3), noise_level=0.01):
    """
    Realistic yet CONTROLLED low-light simulation for colonoscopy images.
    Adjusted parameters to avoid 'gray box' outputs while maintaining realism.
    """
    height, width, _ = image.shape
    degraded_image = image.astype(np.float32)
    # 1. Uneven Illumination (Vignetting) - MUCH MILDER
    center_x, center_y = width // 2, height // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    normalized_dist = dist_from_center / max_dist
    exponent = random.uniform(0.8, 1.2)  # Reduced from (1.2, 1.8)
    vignette_mask = 1 - normalized_dist**exponent
    vignette_mask = np.stack([vignette_mask]*3, axis=-1)
    vignette_scaling = random.uniform(0.7, 0.9)  # Increased from (0.5, 0.8)
    degraded_image *= vignette_mask * vignette_scaling
    # 2. Blur: motion + slight Gaussian - LESS BLUR
    blur_amount = random.randint(*blur_range)
    if blur_amount > 1:
        kernel_size = blur_amount
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        angle = random.choice([0, 45, 90, 135])
        if angle == 0:
            kernel[kernel_size//2, :] = 1
        elif angle == 45:
            for i in range(kernel_size):
                kernel[i, kernel_size - 1 - i] = 1
        elif angle == 90:
            kernel[:, kernel_size//2] = 1
        else:  # 135 degrees
            for i in range(kernel_size):
                kernel[i, i] = 1
        kernel /= kernel.sum()
        degraded_image = cv2.filter2D(degraded_image, -1, kernel)
        degraded_image = cv2.GaussianBlur(degraded_image, (3,3), sigmaX=0.5)
    # 3. Brightness scaling - BRIGHTER
    brightness_factor = random.uniform(*brightness_range)
    degraded_image *= brightness_factor
    # 4. Poisson-like noise - LESS NOISE
    noise = np.random.poisson(degraded_image * noise_level).astype(np.float32)
    degraded_image += noise
    # 5. Slight reddish color cast
    red_scale = random.uniform(1.05, 1.15)
    green_scale = random.uniform(0.95, 1.05)
    blue_scale = random.uniform(0.9, 1.0)
    degraded_image[..., 0] *= blue_scale
    degraded_image[..., 1] *= green_scale
    degraded_image[..., 2] *= red_scale
    # Clip and convert to uint8
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
