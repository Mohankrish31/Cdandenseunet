import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== CBAM =====
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

# ===== Dense Block =====
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(self._make_layer(channels, growth_rate))
            channels += growth_rate
        self.out_channels = channels

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)

# ===== Encoder Block =====
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
        self.cbam = CBAM(self.dense_block.out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.dense_block(x)
        out = self.cbam(out)
        return out, self.pool(out)

# ===== Decoder Block =====
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
        self.cbam = CBAM(self.dense_block.out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        out = self.dense_block(x)
        out = self.cbam(out)
        return out

# ===== CDAN-DenseUNet (Base Channels = 32) =====
class CDANDenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, growth_rate=16, num_layers=3):
        super().__init__()

        # Initial conv to expand input channels to base_channels
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.enc1 = EncoderBlock(base_channels, growth_rate, num_layers)
        self.enc2 = EncoderBlock(self.enc1.dense_block.out_channels, growth_rate, num_layers)
        self.enc3 = EncoderBlock(self.enc2.dense_block.out_channels, growth_rate, num_layers)
        self.enc4 = EncoderBlock(self.enc3.dense_block.out_channels, growth_rate, num_layers)

        # Bottleneck
        self.bottleneck = DenseBlock(self.enc4.dense_block.out_channels, growth_rate, num_layers)
        self.cbam_bottleneck = CBAM(self.bottleneck.out_channels)

        # Decoder
        self.dec4 = DecoderBlock(self.bottleneck.out_channels + self.enc4.dense_block.out_channels, growth_rate, num_layers)
        self.dec3 = DecoderBlock(self.dec4.dense_block.out_channels + self.enc3.dense_block.out_channels, growth_rate, num_layers)
        self.dec2 = DecoderBlock(self.dec3.dense_block.out_channels + self.enc2.dense_block.out_channels, growth_rate, num_layers)
        self.dec1 = DecoderBlock(self.dec2.dense_block.out_channels + self.enc1.dense_block.out_channels, growth_rate, num_layers)

        # Output
        self.final_conv = nn.Conv2d(self.dec1.dense_block.out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)

        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        bottleneck = self.cbam_bottleneck(self.bottleneck(p4))

        d4 = self.dec4(torch.cat([bottleneck, s4], dim=1), s4)
        d3 = self.dec3(torch.cat([d4, s3], dim=1), s3)
        d2 = self.dec2(torch.cat([d3, s2], dim=1), s2)
        d1 = self.dec1(torch.cat([d2, s1], dim=1), s1)

        return torch.sigmoid(self.final_conv(d1))

# ===== Test =====
if __name__ == "__main__":
    model = CDANDenseUNet(in_channels=3, out_channels=3, base_channels=32, growth_rate=16, num_layers=3)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
