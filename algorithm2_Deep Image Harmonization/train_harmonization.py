import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm


# --- 1. 定义一个简单的U-Net作为和谐化模型 ---
# 这是一个非常简化的版本，但足以完成任务并满足“训练模型”的要求
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(True),  # 输入是 3通道合成图 + 1通道蒙版 = 4
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1)  # 输出3通道的调整后图像
        )

    def forward(self, x, mask):
        # 将合成图和蒙版拼接在一起作为输入
        x_with_mask = torch.cat([x, mask], dim=1)
        x = self.encoder(x_with_mask)
        x = self.decoder(x)
        return x


# --- 2. 定义数据集加载器 ---
class HarmonizationDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.composite_dir = os.path.join(data_path, 'composite')
        self.real_dir = os.path.join(data_path, 'real')
        self.mask_dir = os.path.join(data_path, 'mask')
        self.image_names = sorted(os.listdir(self.composite_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        composite_img = Image.open(os.path.join(self.composite_dir, name)).convert('RGB')
        real_img = Image.open(os.path.join(self.real_dir, name)).convert('RGB')
        mask_img = Image.open(os.path.join(self.mask_dir, name)).convert('L')

        if self.transform:
            composite_img = self.transform(composite_img)
            real_img = self.transform(real_img)
            mask_img = self.transform(mask_img)

        return composite_img, real_img, mask_img


# --- 3. 训练主程序 ---
def train():
    # --- 配置参数 ---
    DATA_PATH = "./train_data/"
    MODEL_SAVE_PATH = "./harmonization_model.pth"
    EPOCHS = 10  # 为节省时间，先设置为10轮，您可以根据效果增加
    BATCH_SIZE = 4  # 如果显存不足，可以减小为2或1
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512

    # --- 数据预处理 ---
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # --- 加载数据 ---
    dataset = HarmonizationDataset(DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleUNet().to(device)
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for composite, real, mask in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            composite, real, mask = composite.to(device), real.to(device), mask.to(device)

            optimizer.zero_grad()

            # 模型输出的是调整后的“颜色层”，需要与原图合成得到最终结果
            harmonized_color = model(composite, mask)
            # 只在人像区域应用调整
            harmonized_image = composite * (1 - mask) + harmonized_color * mask

            loss = criterion(harmonized_image, real)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

    # --- 保存模型 ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train()