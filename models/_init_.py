# int.py
import torch
import torch.nn.functional as F
from cdan_denseunet import CDANDenseUNet   # import your model
def main():
    # ------------------- 1. Initialize model -------------------
    model = cdan_denseunet(
        num_classes=2,
        grl_lambda=1.0,
        in_ch=3,
        out_ch=3,
        growth_rate=12,
        block_layers=(3,4,5),
        base_channels=32,
        use_cbam_encoder=True,   # or False if you want plain CDAN-UNet
        use_cbam_decoder=True
    )
    print("✅ cdan_denseunet model initialized.")
    # ------------------- 2. Dummy Input -------------------
    x = torch.randn(8, 3, 224, 224)   # batch_size=8, RGB 224x224 image
    # ------------------- 3. Forward pass -------------------
    gen_out, feat_vec = model(x)
    print(f"Generator output shape: {gen_out.shape}")  # expected (2, 3, 256, 256)
    print(f"Feature vector shape: {feat_vec.shape}")   # expected (2, feat_dim)
    # ------------------- 4. Domain Classifier -------------------
    # Example softmax predictions for CDAN (normally from task head)
    soft_preds = F.softmax(torch.randn(2, 2), dim=1)
    domain_logits = model.domain_classify(feat_vec, soft_preds, grl_lambda=0.5)
    print(f"Domain classifier logits shape: {domain_logits.shape}")  # expected (2,)
    # ------------------- 5. Save Model Weights -------------------
    torch.save(model.state_dict(), "cdan_denseunet.pt")
    print("💾 Model weights saved as cdan_dneseunet.pt")
if __name__ == "__main__":
    main()
