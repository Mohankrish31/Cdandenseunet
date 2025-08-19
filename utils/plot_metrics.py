# plot_metrics.py
import matplotlib.pyplot as plt
def plot_metrics_curve(psnr_list, ssim_list, lpips_list, ebcm_list, save_path=None):
    """
    Plots evaluation metrics (PSNR, SSIM, LPIPS, EBCM) per image.
    Args:
        psnr_list (list): PSNR values per image
        ssim_list (list): SSIM values per image
        lpips_list (list): LPIPS values per image
        ebcm_list (list): EBCM values per image
        save_path (str, optional): if given, saves the plot to this path
    """
    plt.figure(figsize=(10,6))
    plt.plot(psnr_list, label="PSNR", marker='o')
    plt.plot(ssim_list, label="SSIM", marker='s')
    plt.plot(lpips_list, label="LPIPS", marker='^')
    plt.plot(ebcm_list, label="EBCM", marker='d')
    plt.xlabel("Image Index")
    plt.ylabel("Metric Value")
    plt.title("Evaluation Metrics per Image")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Metrics plot saved to {save_path}")
    else:
        plt.show()
    plot_metrics_curve(psnr, ssim, lpips, ebcm, save_path="metrics_curve.png")
