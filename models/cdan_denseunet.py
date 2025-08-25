import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
# ====================== CBAM ======================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
# ================= Dense blocks =================
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else None
    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        if self.dropout:
            out = self.dropout(out)
        return torch.cat([x, out], dim=1)
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer(channels, growth_rate, drop_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels
    def forward(self, x):
        return self.block(x
class Bottleneck(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, drop_rate=0.0):
        super().__init__()
        self.dense_block = _DenseBlock(num_layers, in_channels, growth_rate, drop_rate)
        self.out_channels = self.dense_block.out_channels

    def forward(self, x):
        return self.dense_block(x)

# ============ Encoder & Decoder ============
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers=2, use_cbam=False):
        super().__init__()
        self.db = _DenseBlock(num_layers=num_layers, in_channels=in_channels, growth_rate=growth_rate)
        self.cbam = CBAM(self.db.out_channels) if use_cbam else None
        self.pool = nn.MaxPool2d(2)
    @property
    def out_channels(self):
        return self.db.out_channels
    def forward(self, x):
        x = self.db(x)
        if self.cbam:
            x = self.cbam(x)
        skip = x
        x = self.pool(x)
        return x, skip
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, num_layers=2, use_cbam=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.db = _DenseBlock(num_layers=num_layers,
                              in_channels=2 * skip_channels,
                              growth_rate=skip_channels // 2)
        out_channels = 2 * skip_channels + num_layers * (skip_channels // 2)
        self.compress = nn.Conv2d(out_channels, skip_channels, kernel_size=1, bias=False)
        self.cbam = CBAM(skip_channels) if use_cbam else None
        self._out_channels = skip_channels
    @property
    def out_channels(self):
        return self._out_channels
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[:, :, dh // 2: skip.shape[-2] - (dh - dh // 2),
                              dw // 2: skip.shape[-1] - (dw - dw // 2)]
        x = torch.cat([x, skip], dim=1)
        x = self.db(x)
        x = self.compress(x)
        if self.cbam:
            x = self.cbam(x)
        return x
# ================= CDAN-DenseUNet =================
class CDANDenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=24, growth_rate=12, output_range="01"):
        """
        output_range: "01" → outputs in [0,1], use Sigmoid
                      "11" → outputs in [-1,1], use Tanh
        """
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # ---- Encoders ----
        self.enc1 = EncoderBlock(base_channels, growth_rate, num_layers=2, use_cbam=False)
        self.enc2 = EncoderBlock(self.enc1.out_channels, growth_rate, num_layers=2, use_cbam=False)
        self.enc3 = EncoderBlock(self.enc2.out_channels, growth_rate, num_layers=2, use_cbam=True)
        # ---- Bottleneck ----
        self.bottleneck = Bottleneck(self.enc3.out_channels, num_layers=2, growth_rate=growth_rate)
        self.cbam_bottleneck = CBAM(self.bottleneck.out_channels)
        # ---- Decoders ----
        self.dec3 = DecoderBlock(self.bottleneck.out_channels, self.enc3.out_channels, num_layers=2, use_cbam=False)
        self.dec2 = DecoderBlock(self.enc3.out_channels, self.enc2.out_channels, num_layers=2, use_cbam=False)
        self.dec1 = DecoderBlock(self.enc2.out_channels, self.enc1.out_channels, num_layers=2, use_cbam=True)
        # ---- Final conv + activation ----
        self.final = nn.Conv2d(self.enc1.out_channels, out_channels, kernel_size=1)
        if output_range == "01":
            self.final_activation = nn.Sigmoid()   # ✅ outputs in [0,1]
        elif output_range == "11":
            self.final_activation = nn.Tanh()      # ✅ outputs in [-1,1]
        else:
            raise ValueError("output_range must be '01' or '11'")
    def forward(self, x):
        x = self.init_conv(x)
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x = self.bottleneck(x)
        x = self.cbam_bottleneck(x)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        out = self.final_activation(self.final(x))
        return out
# ===================== Debug + Save =====================
def save_tensor_image(tensor, filename):
    """Save a model output tensor as an image"""
    img = tensor.detach().cpu().permute(1,2,0).numpy()  # [H,W,C]
    img = (img * 255).clip(0,255).astype(np.uint8)
    Image.fromarray(img).save(filename)
if __name__ == "__main__":
    # Example: outputs in [0,1]
    model = CDANDenseUNet(in_channels=3, out_channels=3, base_channels=32, growth_rate=12, output_range="01")
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    print(f"Range → min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}")
    save_tensor_image(y[0], "test_output.png")
    print("Saved: test_output.png")
