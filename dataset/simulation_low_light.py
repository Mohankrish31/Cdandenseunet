import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
class cvccolondbsplitDataset(Dataset):
    def __init__(self, low_root, high_root, transform=None, simulate=False):
        self.low_root = low_root
        self.high_root = high_root
        self.transform = transform
        self.simulate = simulate
        self.to_tensor = transforms.ToTensor()
        self.filenames = sorted(os.listdir(high_root))
    def simulate_low_light(self, image):
        np_img = np.array(image).astype(np.float32) / 255.0
        gamma = random.uniform(1.5, 2.5)
        low_light_img = np.power(np_img, gamma)
        poisson_scaled = low_light_img * random.uniform(20, 50)
        poisson_noise = np.random.poisson(poisson_scaled) / random.uniform(20, 50)
        noisy_img = np.clip(low_light_img + poisson_noise, 0, 1)
        mean, std = 0, random.uniform(0.01, 0.05)
        noisy_img = np.clip(noisy_img + np.random.normal(mean, std, noisy_img.shape), 0, 1)
        jitter = transforms.ColorJitter(
            brightness=random.uniform(0.1, 0.3),
            contrast=random.uniform(0.0, 0.1),
            saturation=random.uniform(0.0, 0.1),
            hue=random.uniform(-0.05, 0.05)
        )
        pil_img = Image.fromarray((noisy_img * 255).astype(np.uint8))
        jittered_img = jitter(pil_img)
        return jittered_img
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        high_img_path = os.path.join(self.high_root, self.filenames[idx])
        high = Image.open(high_img_path).convert('RGB')
        if self.simulate:
            low = self.simulate_low_light(high.copy())
        else:
            low_img_path = os.path.join(self.low_root, self.filenames[idx])
            low = Image.open(low_img_path).convert('RGB')
        # ✅ Apply SAME transform to both
        if self.transform:
            seed = np.random.randint(2147483647)  # ensure same randomness
            random.seed(seed); torch.manual_seed(seed)
            low = self.transform(low)
            random.seed(seed); torch.manual_seed(seed)
            high = self.transform(high)
        else:
            low = self.to_tensor(low)
            high = self.to_tensor(high)
        return low, high
