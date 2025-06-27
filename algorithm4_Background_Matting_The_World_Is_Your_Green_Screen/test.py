import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# 定义上下文切换块 (Context Switching Block)
class ContextSwitchingBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=64):
        super(ContextSwitchingBlock, self).__init__()
        self.selector = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, image_feat, other_feat):
        combined = torch.cat([image_feat, other_feat], dim=1)
        return self.selector(combined)


# 定义完整的背景抠图网络
class BackgroundMattingNetwork(nn.Module):
    def __init__(self):
        super(BackgroundMattingNetwork, self).__init__()

        # 编码器部分
        self.image_encoder = self._make_encoder(3)  # 输入3通道
        self.background_encoder = self._make_encoder(3)  # 输入3通道
        self.mask_encoder = self._make_encoder(1)  # 输入1通道

        # 上下文切换块
        self.cs_block_bg = ContextSwitchingBlock()
        self.cs_block_mask = ContextSwitchingBlock()

        # 组合器
        self.combinator = nn.Sequential(
            nn.Conv2d(256 + 64 + 64, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 解码器部分
        self.decoder = self._make_decoder()

        # 输出层
        self.alpha_output = nn.Conv2d(64, 1, kernel_size=1)
        self.foreground_output = nn.Conv2d(64, 3, kernel_size=1)

    def _make_encoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _make_decoder(self):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, image, background, mask):
        # 编码特征
        image_feat = self.image_encoder(image)
        bg_feat = self.background_encoder(background)
        mask_feat = self.mask_encoder(mask)

        # 上下文切换
        bg_switched = self.cs_block_bg(image_feat, bg_feat)
        mask_switched = self.cs_block_mask(image_feat, mask_feat)

        # 组合特征
        combined = torch.cat([image_feat, bg_switched, mask_switched], dim=1)
        combined = self.combinator(combined)

        # 解码
        decoded = self.decoder(combined)

        # 输出
        alpha = torch.sigmoid(self.alpha_output(decoded))
        foreground = torch.sigmoid(self.foreground_output(decoded))

        return foreground, alpha



# 加载训练好的模型
def load_model(model_path):
    model = BackgroundMattingNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 测试数据集类
class TestMattingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        self.image_files = sorted(glob.glob(os.path.join(root_dir, 'images', '*.png')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')
        background = Image.open(os.path.join(self.root_dir, 'backgrounds', base_name)).convert('RGB')
        mask = Image.open(os.path.join(self.root_dir, 'masks', base_name)).convert('L')
        portrait = Image.open(os.path.join(self.root_dir, 'portraits', base_name)).convert('RGB')

        # 应用转换
        image = self.transform(image)
        background = self.transform(background)
        mask = transforms.ToTensor()(mask)
        portrait = self.transform(portrait)

        return {
            'image': image,
            'background': background,
            'mask': mask,
            'portrait': portrait,
            'name': base_name
        }


def compute_metrics(pred_fg, pred_alpha, true_fg, true_alpha):
    # 转换为numpy数组
    pred_fg_np = pred_fg.squeeze().permute(1, 2, 0).cpu().numpy()
    pred_alpha_np = pred_alpha.squeeze().cpu().numpy()
    true_fg_np = true_fg.squeeze().permute(1, 2, 0).cpu().numpy()
    true_alpha_np = true_alpha.squeeze().cpu().numpy()

    # 计算PSNR
    fg_psnr = psnr(true_fg_np, pred_fg_np, data_range=1.0)
    alpha_psnr = psnr(true_alpha_np, pred_alpha_np, data_range=1.0)

    # 计算SSIM - 添加win_size参数
    min_dim = min(pred_fg_np.shape[0], pred_fg_np.shape[1])
    win_size = min(7, min_dim)  # 确保不超过图像尺寸
    win_size = win_size if win_size % 2 == 1 else win_size - 1  # 确保是奇数

    try:
        fg_ssim = ssim(true_fg_np, pred_fg_np,
                       win_size=win_size,
                       channel_axis=2,  # 替换旧的multichannel参数
                       data_range=1.0)

        alpha_ssim = ssim(true_alpha_np, pred_alpha_np,
                          win_size=win_size,
                          data_range=1.0)
    except ValueError as e:
        print(f"SSIM计算错误: {e}")
        fg_ssim = 0.0
        alpha_ssim = 0.0

    # 计算MSE
    fg_mse = np.mean((true_fg_np - pred_fg_np) ** 2)
    alpha_mse = np.mean((true_alpha_np - pred_alpha_np) ** 2)

    return {
        'fg_psnr': fg_psnr,
        'alpha_psnr': alpha_psnr,
        'fg_ssim': fg_ssim,
        'alpha_ssim': alpha_ssim,
        'fg_mse': fg_mse,
        'alpha_mse': alpha_mse
    }


# 可视化结果
def visualize_results(image, background, mask, pred_fg, pred_alpha, composite, save_path):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 6, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.imshow(background.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title('Background')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Input Mask')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.imshow(pred_fg.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title('Predicted FG')
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.imshow(pred_alpha.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Predicted Alpha')
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(composite.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title('Composite Result')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


# 主测试函数
def test_model(model_path, test_data_dir, output_dir='test_results'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path).to(device)

    # 加载测试数据
    test_dataset = TestMattingDataset(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化指标存储
    all_metrics = {
        'fg_psnr': [],
        'alpha_psnr': [],
        'fg_ssim': [],
        'alpha_ssim': [],
        'fg_mse': [],
        'alpha_mse': []
    }

    print(f"开始测试，共 {len(test_dataset)} 个样本...")

    for i, batch in enumerate(test_loader):
        # 准备数据
        image = batch['image'].to(device)
        background = batch['background'].to(device)
        mask = batch['mask'].to(device)
        true_fg = batch['portrait'].to(device)
        true_alpha = mask
        name = batch['name'][0]

        # 前向传播
        with torch.no_grad():
            pred_fg, pred_alpha = model(image, background, mask)

        # 合成图像
        composite = pred_fg * pred_alpha + background * (1 - pred_alpha)

        # 计算指标
        metrics = compute_metrics(pred_fg, pred_alpha, true_fg, true_alpha)

        # 存储指标
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        # 可视化结果（每5个样本保存一个）
        if i % 5 == 0:
            save_path = os.path.join(output_dir, f'result_{name[:-4]}.png')  # 移除.png后缀
            visualize_results(image, background, mask, pred_fg, pred_alpha, composite, save_path)

        print(f"样本 {i + 1}/{len(test_dataset)} - {name}")
        print(f"  FG PSNR: {metrics['fg_psnr']:.2f}, Alpha PSNR: {metrics['alpha_psnr']:.2f}")
        print(f"  FG SSIM: {metrics['fg_ssim']:.4f}, Alpha SSIM: {metrics['alpha_ssim']:.4f}")

    # 计算平均指标
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    print("\n测试完成，平均指标:")
    print(f"前景 PSNR: {avg_metrics['fg_psnr']:.2f}")
    print(f"Alpha PSNR: {avg_metrics['alpha_psnr']:.2f}")
    print(f"前景 SSIM: {avg_metrics['fg_ssim']:.4f}")
    print(f"Alpha SSIM: {avg_metrics['alpha_ssim']:.4f}")
    print(f"前景 MSE: {avg_metrics['fg_mse']:.6f}")
    print(f"Alpha MSE: {avg_metrics['alpha_mse']:.6f}")

    # 保存指标到文件
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("平均测试指标:\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    return avg_metrics


if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = 'background_matting_model.pth'
    TEST_DATA_DIR = 'test_dataset'
    OUTPUT_DIR = 'test_results'

    # 运行测试
    test_model(MODEL_PATH, TEST_DATA_DIR, OUTPUT_DIR)