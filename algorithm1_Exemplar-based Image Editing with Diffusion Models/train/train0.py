import torch.optim as optim
from tqdm import tqdm
from Diffusion import *
from dataset import *
# 自动选择设备
import torch
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.version.cuda)  # 输出 PyTorch 支持的 CUDA 版本
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 初始化扩散模型
model = DiffusionModel().to(device)

# 确保模型参数是 float32 类型
model = model.float()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练过程
num_epochs = 50

for epoch in range(num_epochs):
    for i, (source_image, exemplar_image, mask, background_image) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        source_image = source_image.to(device).float()
        exemplar_image = exemplar_image.to(device).float()
        mask = mask.to(device).float()
        background_image = background_image.to(device).float()

        # 前向传播
        optimizer.zero_grad()
        output = model(exemplar_image * mask + background_image * (1 - mask))
        loss = criterion(output, source_image)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "D:/BaiduSyncdisk/streetphoto/Street_photography/models/diffmodel.pth")