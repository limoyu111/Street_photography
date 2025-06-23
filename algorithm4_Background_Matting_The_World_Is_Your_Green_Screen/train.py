import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt


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


# 自定义数据集类
class MattingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(root_dir, 'images', '*.png')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        background = Image.open(os.path.join(self.root_dir, 'backgrounds', base_name)).convert('RGB')
        mask = Image.open(os.path.join(self.root_dir, 'masks', base_name)).convert('L')  # 保持为单通道
        portrait = Image.open(os.path.join(self.root_dir, 'portraits', base_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)
            background = self.transform(background)
            # 对掩码特殊处理，保持单通道
            mask = transforms.ToTensor()(mask)
            portrait = self.transform(portrait)

        return {
            'image': image,
            'background': background,
            'mask': mask,
            'portrait': portrait
        }


# 定义损失函数
class MattingLoss(nn.Module):
    def __init__(self):
        super(MattingLoss, self).__init__()

    def forward(self, pred_fg, pred_alpha, true_fg, true_alpha, image, background):
        # Alpha损失
        alpha_loss = torch.abs(pred_alpha - true_alpha).mean()

        # Alpha梯度损失
        grad_true = torch.abs(true_alpha[:, :, :-1, :-1] - true_alpha[:, :, 1:, :-1]) + \
                    torch.abs(true_alpha[:, :, :-1, :-1] - true_alpha[:, :, :-1, 1:])
        grad_pred = torch.abs(pred_alpha[:, :, :-1, :-1] - pred_alpha[:, :, 1:, :-1]) + \
                    torch.abs(pred_alpha[:, :, :-1, :-1] - pred_alpha[:, :, :-1, 1:])
        alpha_grad_loss = torch.abs(grad_pred - grad_true).mean()

        # 前景损失
        fg_loss = torch.abs(pred_fg - true_fg).mean()

        # 合成损失
        comp_loss = torch.abs(image - pred_alpha * pred_fg - (1 - pred_alpha) * background).mean()

        # 总损失
        total_loss = alpha_loss + 2 * alpha_grad_loss + fg_loss + comp_loss

        return total_loss


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            # 获取数据
            image = batch['image'].to(device)
            background = batch['background'].to(device)
            mask = batch['mask'].to(device)
            true_fg = batch['portrait'].to(device)

            # 创建真实alpha (从mask)
            true_alpha = mask  # 直接使用单通道mask作为alpha

            # 前向传播
            optimizer.zero_grad()
            pred_fg, pred_alpha = model(image, background, mask)

            # 计算损失
            loss = criterion(pred_fg, pred_alpha, true_fg, true_alpha, image, background)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


# 图像合成函数
def composite_image(foreground, alpha, background):
    """ 将前景与背景按照alpha通道合成 """
    composite = foreground * alpha + background * (1 - alpha)
    return composite


# 主函数
def main():
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # 创建数据集和数据加载器
    dataset = MattingDataset(root_dir='dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # 初始化模型
    model = BackgroundMattingNetwork()

    # 定义损失函数和优化器
    criterion = MattingLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    print("开始训练...")
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=20)

    # 保存模型
    torch.save(trained_model.state_dict(), 'background_matting_model.pth')
    print("模型已保存为 background_matting_model.pth")

    # 测试合成效果
    test_sample = dataset[0]
    image = test_sample['image'].unsqueeze(0)
    background = test_sample['background'].unsqueeze(0)
    mask = test_sample['mask'].unsqueeze(0)

    with torch.no_grad():
        trained_model.eval()
        pred_fg, pred_alpha = trained_model(image, background, mask)

    # 合成图像
    composite = composite_image(pred_fg, pred_alpha, background)

    # 可视化结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title('原始图像')

    plt.subplot(1, 4, 2)
    plt.imshow(pred_fg.squeeze().permute(1, 2, 0))
    plt.title('预测前景')

    plt.subplot(1, 4, 3)
    plt.imshow(pred_alpha.squeeze(), cmap='gray')
    plt.title('预测Alpha')

    plt.subplot(1, 4, 4)
    plt.imshow(composite.squeeze().permute(1, 2, 0))
    plt.title('合成结果')

    plt.tight_layout()
    plt.savefig('result.png')
    print("测试结果已保存为 result.png")


if __name__ == '__main__':
    main()