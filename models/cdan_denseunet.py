import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# ------------------------ CBAM ------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.avg_pool(x)
        m = self.max_pool(x)
        out = self.fc(a) + self.fc(m)
        return self.sigmoid(out) * x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, maxv], dim=1)
        map = self.conv(cat)
        return self.sigmoid(map) * x
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
# ------------------------ Dense Block & Transition ------------------------
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate
    def forward(self, x):
        new_features = self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(x))))))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layer = _DenseLayer(channels, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )
    def forward(self, x):
        return self.net(x)
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Using ConvTranspose2d for upsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)
# ------------------------ MultiScale Pool ------------------------
class MultiScalePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # simple multi-scale feature aggregator
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b1 = F.interpolate(self.pool1(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        b2 = F.interpolate(self.pool2(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        b3 = F.interpolate(self.pool3(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        cat = torch.cat([b1, b2, b3], dim=1)
        out = self.relu(self.conv(cat))
        return out
# ------------------------ Gradient Reversal Layer ------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None
class GradReverse(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
# ------------------------ CDAN Discriminator ------------------------
class CDANDiscriminator(nn.Module):
    def __init__(self, feat_dim, class_dim, hidden=1024):
        super().__init__()
        # we'll handle conditioning via outer product (feature x softmax)
        # but to keep it efficient we'll project both to vectors and then multiply
        self.feature_proj = nn.Linear(feat_dim, hidden)
        self.class_proj = nn.Linear(class_dim, hidden)
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, features, soft_preds):
        # features: [B, feat_dim]
        # soft_preds: [B, class_dim]
        f = self.feature_proj(features)
        c = self.class_proj(soft_preds)
        # element-wise multiply (conditioning)
        h = f * c
        out = self.net(h)
        return out.squeeze(1)
# ------------------------ DenseUNet with optional CBAM ------------------------
class DenseUNetCBAM(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, growth_rate=16, block_layers=(4,5,7,10),
                 bn_size=4, drop_rate=0, base_channels=48, use_cbam_encoder=False, use_cbam_decoder=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.growth_rate = growth_rate
        self.block_layers = block_layers
        self.base_channels = base_channels
        # Initial conv
        self.init_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1, bias=False)
        # Encoder: DenseBlock -> TransitionDown
        enc_channels = base_channels
        self.enc_blocks = nn.ModuleList()
        self.trans_downs = nn.ModuleList()
        self.cbam_enc = nn.ModuleList() if use_cbam_encoder else None
        for num_layers in block_layers:
            db = DenseBlock(num_layers, enc_channels, growth_rate, bn_size, drop_rate)
            self.enc_blocks.append(db)
            enc_channels = db.out_channels
            if use_cbam_encoder:
                self.cbam_enc.append(CBAM(enc_channels))
            # reduce channels for next stage
            td = TransitionDown(enc_channels, enc_channels // 2)
            self.trans_downs.append(td)
            enc_channels = enc_channels // 2
        # Bottleneck
        self.bottleneck = DenseBlock(num_layers=block_layers[-1], in_channels=enc_channels, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        bottleneck_channels = self.bottleneck.out_channels
        # Decoder: TransitionUp -> concat skip -> DenseBlock
        self.trans_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.cbam_dec = nn.ModuleList() if use_cbam_decoder else None
        dec_channels = bottleneck_channels
        # build decoder in reverse
        for i, num_layers in enumerate(reversed(block_layers)):
            tu = TransitionUp(dec_channels, dec_channels // 2)
            self.trans_ups.append(tu)
            dec_channels = dec_channels // 2
            # after up, we'll concatenate skip -> channels grow
            # assume skip_channels roughly = enc_blocks[i].out_channels // 2 (because of transition down earlier)
            skip_ch = self.enc_blocks[-(i+1)].out_channels // 2
            in_ch_for_db = dec_channels + skip_ch
            db = DenseBlock(num_layers, in_ch_for_db, growth_rate, bn_size, drop_rate)
            self.dec_blocks.append(db)
            dec_channels = db.out_channels
            if use_cbam_decoder:
                self.cbam_dec.append(CBAM(dec_channels))
        # Final conv to output
        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=1)
        )
    def forward(self, x):
        skips = []
        out = self.init_conv(x)
        # Encoder
        for i, db in enumerate(self.enc_blocks):
            out = db(out)
            if self.cbam_enc is not None:
                out = self.cbam_enc[i](out)
            skips.append(out)
            out = self.trans_downs[i](out)
        # Bottleneck
        out = self.bottleneck(out)
        # Decoder
        for i, tu in enumerate(self.trans_ups):
            out = tu(out)
            # get corresponding skip (reverse)
            skip = skips[-(i+1)]
            # make sure sizes match (crop/pad if necessary)
            if out.size(2) != skip.size(2) or out.size(3) != skip.size(3):
                out = F.interpolate(out, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = self.dec_blocks[i](out)
            if self.cbam_dec is not None:
                out = self.cbam_dec[i](out)
        out = self.final_conv(out)
        return out
# ------------------------ CDAN DenseUNet Wrapper ------------------------
class CDANDenseUNet(nn.Module):
    def __init__(self, num_classes=2, grl_lambda=1.0, **gen_kwargs):
        """gen_kwargs forwarded to DenseUNetCBAM"""
        super().__init__()
        self.generator = DenseUNetCBAM(**gen_kwargs)
        # we will extract a pooled feature vector from a chosen layer (e.g., bottleneck output)
        # To keep simple, add a global pooling on top of generator's penultimate feature map.
        # Option: you can modify DenseUNetCBAM to return intermediate features explicitly.
        # Here we'll add a small feature extractor that expects bottleneck-level features.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # projection dimension depends on channel count: estimate by running a dummy tensor in practice
        # For flexibility, we'll compute feature projection on the fly in forward (lazy way)
        self.num_classes = num_classes
        self.grl = GradReverse(lambda_=grl_lambda)
        # discriminator will be initialized lazily once we know feat dim
        self.discriminator = None
    def forward(self, x):
        # forward through generator, but we need access to an intermediate "feature map" for CDAN.
        # For simplicity we will hack into generator: run encoder until bottleneck then use pool on bottleneck
        out = x
        out = self.generator.init_conv(out)
        skips = []
        for i, db in enumerate(self.generator.enc_blocks):
            out = db(out)
            if self.generator.cbam_enc is not None:
                out = self.generator.cbam_enc[i](out)
            skips.append(out)
            out = self.generator.trans_downs[i](out)
        # bottleneck feature map
        bottleneck_feat = self.generator.bottleneck(out)
        # generator output continues through decoder to produce enhanced image
        dec = bottleneck_feat
        for i, tu in enumerate(self.generator.trans_ups):
            dec = tu(dec)
            skip = skips[-(i+1)]
            if dec.size(2) != skip.size(2) or dec.size(3) != skip.size(3):
                dec = F.interpolate(dec, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
            dec = torch.cat([dec, skip], dim=1)
            dec = self.generator.dec_blocks[i](dec)
            if self.generator.cbam_dec is not None:
                dec = self.generator.cbam_dec[i](dec)
        gen_out = self.generator.final_conv(dec)
        # Feature vector for discriminator
        pooled = self.pool(bottleneck_feat)
        feat_vec = pooled.view(pooled.size(0), -1)  # [B, feat_dim]
        return gen_out, feat_vec
    def init_discriminator(self, feat_dim, class_dim, hidden=1024):
        self.discriminator = CDANDiscriminator(feat_dim, class_dim, hidden)
    def domain_classify(self, feat_vec, soft_preds, grl_lambda=None):
        """
        feat_vec: [B, feat_dim]
        soft_preds: [B, num_classes]  (softmax probabilities)
        Returns discriminator logits (before sigmoid)
        """
        if self.discriminator is None:
            # lazily init
            self.init_discriminator(feat_vec.size(1), soft_preds.size(1))
        if grl_lambda is not None:
            # temporarily change grl lambda
            self.grl.lambda_ = grl_lambda
        feat_rev = self.grl(feat_vec)
        logits = self.discriminator(feat_rev, soft_preds)
        return logits
# ------------------------ Example usage helper ------------------------
if __name__ == '__main__':
    # quick smoke test
    model = CDANDenseUNet(num_classes=2, grl_lambda=1.0,
                          in_ch=3, out_ch=3, growth_rate=12, block_layers=(3,4,5),
                          base_channels=32, use_cbam_encoder=True, use_cbam_decoder=True)
    x = torch.randn(2,3,256,256)
    gen_out, feat = model(x)
    print('gen_out', gen_out.shape)
    print('feat', feat.shape)
    # fake soft preds
    soft = F.softmax(torch.randn(2,2), dim=1)
    logits = model.domain_classify(feat, soft, grl_lambda=0.5)
    print('logits', logits.shape)
# End of file
