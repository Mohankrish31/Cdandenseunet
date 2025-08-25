import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
# Add your model folder to the path
sys.path.append('/content/CdanDenseUNet')
from models.cdan_denseunet import CDANDenseUNet
# ------------------- Dataset Class -------------------
class LowLightDataset(Dataset):
    def __init__(self, low_root, high_root, transform=None):
        self.low_root = low_root
        self.high_root = high_root
        self.transform = transform
        self.filenames = sorted(os.listdir(low_root))
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        low_path = os.path.join(self.low_root, self.filenames[idx])
        high_path = os.path.join(self.high_root, self.filenames[idx])
        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        return low_img, high_img
# ------------------- Training Script -------------------
def train_model():
    # Paths
    dataset_root = "/content/cvccolondbsplit"
    train_low_dir = os.path.join(dataset_root, "train", "low")
    train_high_dir = os.path.join(dataset_root, "train", "high")
    model_save_path = "/content/saved_model/cdan_denseunet_isp_weights.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # Hyperparameters
    num_epochs = 100
    learning_rate = 0.0001
    batch_size = 8
    weight_decay = 0.0001  # Added weight decay parameter
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data transformations (normalize to [0,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Datasets and DataLoaders
    train_dataset = LowLightDataset(low_root=train_low_dir, high_root=train_high_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Model, Optimizer, Loss Function
    model = CDANDenseUNet(in_channels=3, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Included weight_decay
    criterion = torch.nn.MSELoss()
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for i, (low_img, high_img) in enumerate(train_loader):
            low_img = low_img.to(device)
            high_img = high_img.to(device)
            # Forward pass
            enhanced_img = model(low_img)
            loss = criterion(enhanced_img, high_img)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    # Save the trained model weights
    torch.save(model.state_dict(), model_save_path)
    print("✅ Model trained and weights saved successfully!")
if __name__ == "__main__":
    train_model()
