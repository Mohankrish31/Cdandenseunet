import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------ CBAM ------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ------------------------ Dense Block ------------------------
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ------------------------ Encoder & Decoder ------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers=2, use_cbam=False):
        super().__init__()
        self.db = DenseBlock(in_channels, growth_rate, num_layers)
        self.cbam = CBAM(in_channels + num_layers * growth_rate) if use_cbam else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.db(x)
        if self.cbam:
            x = self.cbam(x)
        skip = x
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.db = DenseBlock(out_channels * 2, growth_rate=out_channels // 2, num_layers=2)
        self.cbam = CBAM(out_channels * 2) if use_cbam else None

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.db(x)
        if self.cbam:
            x = self.cbam(x)
        return x

# ------------------------ CDAN DenseUNet (Light) ------------------------
class CDANDenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=24, growth_rate=12):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder: only 3
        self.enc1 = EncoderBlock(base_channels, growth_rate, num_layers=2, use_cbam=False)
        self.enc2 = EncoderBlock(base_channels + 2 * growth_rate, growth_rate, num_layers=2, use_cbam=False)
        self.enc3 = EncoderBlock(base_channels + 4 * growth_rate, growth_rate, num_layers=2, use_cbam=True)  # CBAM only here

        # Bottleneck
        bottleneck_in = base_channels + 6 * growth_rate
        self.bottleneck = DenseBlock(bottleneck_in, growth_rate, num_layers=2)
        self.cbam_bottleneck = CBAM(bottleneck_in + 2 * growth_rate)

        # Decoder: only 3
        self.dec3 = DecoderBlock(bottleneck_in + 2 * growth_rate, base_channels + 4 * growth_rate, use_cbam=False)
        self.dec2 = DecoderBlock(base_channels + 4 * growth_rate, base_channels + 2 * growth_rate, use_cbam=False)
        self.dec1 = DecoderBlock(base_channels + 2 * growth_rate, base_channels, use_cbam=True)  # CBAM only here

        # Final
        self.final = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1)

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

        out = self.final(x)
        return out

# ------------------------ Test ------------------------
if __name__ == "__main__":
    model = CDANDenseUNet()
    dummy = torch.randn(8, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
