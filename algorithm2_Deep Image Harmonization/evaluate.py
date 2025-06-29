import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips


# --- 1. 重新定义模型结构 (必须与训练时完全一致) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x, mask):
        x_with_mask = torch.cat([x, mask], dim=1)
        x = self.encoder(x_with_mask)
        x = self.decoder(x)
        return x


def evaluate():
    # --- 配置 ---
    TEST_DATA_PATH = "./test_data/"
    MODEL_PATH = "./harmonization_model.pth"
    IMAGE_SIZE = 512

    # --- 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"模型已从 {MODEL_PATH} 加载，使用设备: {device}")

    # --- 新增：加载 LPIPS 评测模型 ---
    print("正在加载 LPIPS 感知相似度评测模型...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # --- 数据转换 ---
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # --- 初始化四个指标的累加器 ---
    total_psnr, total_ssim, total_mse, total_lpips = 0, 0, 0, 0
    image_list = sorted(os.listdir(os.path.join(TEST_DATA_PATH, 'composite')))
    num_images = len(image_list)

    if num_images == 0:
        print(f"错误：在 '{TEST_DATA_PATH}' 中没有找到测试图片。")
        return

    # --- 评测循环 ---
    with torch.no_grad():
        for name in tqdm(image_list, desc="正在评测测试集"):
            composite_path = os.path.join(TEST_DATA_PATH, 'composite', name)
            real_path = os.path.join(TEST_DATA_PATH, 'real', name)
            mask_path = os.path.join(TEST_DATA_PATH, 'mask', name)

            composite_pil = Image.open(composite_path).convert('RGB')
            real_pil = Image.open(real_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('L')

            # 准备模型输入张量
            composite_tensor = transform(composite_pil).unsqueeze(0).to(device)
            mask_tensor = transform(mask_pil).unsqueeze(0).to(device)
            real_tensor = transform(real_pil).unsqueeze(0).to(device)

            # 模型推理
            harmonized_color = model(composite_tensor, mask_tensor)
            harmonized_tensor = composite_tensor * (1 - mask_tensor) + harmonized_color * mask_tensor

            # --- 开始计算四项指标 ---

            # A. 准备用于 PSNR, SSIM, MSE 的Numpy数组 (范围 0-1)
            harmonized_np = harmonized_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            real_np = real_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            harmonized_np = np.clip(harmonized_np, 0, 1)
            real_np = np.clip(real_np, 0, 1)

            # 计算 PSNR, SSIM, MSE
            total_psnr += psnr(real_np, harmonized_np, data_range=1)
            total_ssim += ssim(real_np, harmonized_np, channel_axis=2, data_range=1)
            total_mse += np.mean((real_np - harmonized_np) ** 2)

            # B. 准备用于 LPIPS 的Tensor (范围 -1 到 1)
            harmonized_lpips = harmonized_tensor * 2 - 1
            real_lpips = real_tensor * 2 - 1
            total_lpips += lpips_fn(harmonized_lpips, real_lpips).item()

    # --- 计算并打印最终平均分 ---
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_mse = total_mse / num_images
    avg_lpips = total_lpips / num_images

    print("\n--- 评测完成 ---")
    print(f"测试集图片总数: {num_images}")
    print(f"平均 PSNR (越高越好): {avg_psnr:.4f}")
    print(f"平均 SSIM (越高越好): {avg_ssim:.4f}")
    print(f"平均 MSE (越低越好):  {avg_mse:.4f}")
    print(f"平均 LPIPS (越低越好): {avg_lpips:.4f}")


if __name__ == '__main__':
    evaluate()