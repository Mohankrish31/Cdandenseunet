import torch 
from torch.utils.data import Dataset, DataLoader #Corrected import 
import os 
from PIL import Image 
from torchvision import transforms 
# === Dataset ===
class cvccolondbsplitDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg')) and
               os.path.exists(os.path.join(high_dir, f))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.image_names[idx])
        high_path = os.path.join(self.high_dir, self.image_names[idx])

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img   # <-- input=low, target=high

# === Define Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# === Paths (example usage) ===
train_high_dir = "/content/cvccolondbsplit/train/high"
train_low_dir = "/content/cvccolondbsplit/train/low"
val_high_dir = "/content/cvccolondbsplit/val/high"
val_low_dir = "/content/cvccolondbsplit/val/low"
# === Create Dataset Instances ===
train_dataset = cvccolondbsplitDataset(train_low_dir, train_high_dir, transform=transform)
val_dataset = cvccolondbsplitDataset(val_low_dir, val_high_dir, transform=transform)
# === Create DataLoaders ===
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
