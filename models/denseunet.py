import torch
import torch.nn as nn
import torch.nn.functional as F
# === Dense Block ===
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(self._make_layer(channels, growth_rate))
            channels += growth_rate
        self.out_channels = channels
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)
# === DenseUNet Encoder Block ===
class DenseUNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseUNetEncoderBlock, self).__init__()
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        out = self.dense_block(x)
        return out, self.pool(out)
# === DenseUNet Decoder Block ===
class DenseUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseUNetDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.dense_block(x)
# === DenseUNet ===
class denseunet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, growth_rate=16, num_layers=3):
        super(DenseUNet, self).__init__()
        # Encoder
        self.enc1 = DenseUNetEncoderBlock(in_channels, growth_rate, num_layers)
        self.enc2 = DenseUNetEncoderBlock(self.enc1.dense_block.out_channels, growth_rate, num_layers)
        self.enc3 = DenseUNetEncoderBlock(self.enc2.dense_block.out_channels, growth_rate, num_layers)
        self.enc4 = DenseUNetEncoderBlock(self.enc3.dense_block.out_channels, growth_rate, num_layers)
        # Bottleneck
        self.bottleneck = DenseBlock(self.enc4.dense_block.out_channels, growth_rate, num_layers)
        # Decoder
        self.dec4 = DenseUNetDecoderBlock(self.bottleneck.out_channels + self.enc4.dense_block.out_channels, growth_rate, num_layers)
        self.dec3 = DenseUNetDecoderBlock(self.dec4.dense_block.out_channels + self.enc3.dense_block.out_channels, growth_rate, num_layers)
        self.dec2 = DenseUNetDecoderBlock(self.dec3.dense_block.out_channels + self.enc2.dense_block.out_channels, growth_rate, num_layers)
        self.dec1 = DenseUNetDecoderBlock(self.dec2.dense_block.out_channels + self.enc1.dense_block.out_channels, growth_rate, num_layers)
        # Output layer
        self.final_conv = nn.Conv2d(self.dec1.dense_block.out_channels, out_channels, kernel_size=1)
    def forward(self, x):
        # Encoder
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        # Decoder
        d4 = self.dec4(torch.cat([bottleneck, s4], dim=1), s4)
        d3 = self.dec3(torch.cat([d4, s3], dim=1), s3)
        d2 = self.dec2(torch.cat([d3, s2], dim=1), s2)
        d1 = self.dec1(torch.cat([d2, s1], dim=1), s1)
        return torch.sigmoid(self.final_conv(d1))


if __name__ == "__main__":
    model = DenseUNet(in_channels=3, out_channels=3, growth_rate=16, num_layers=3)
    x = torch.randn(1, 3, 224, 224)  # For training phase
    y = model(x)
    print("Output shape:", y.shape)
