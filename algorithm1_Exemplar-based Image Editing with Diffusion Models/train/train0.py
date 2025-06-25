import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
import clip
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# 加载预训练的 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class StreetSnapDataset(Dataset):
    def __init__(self, source_images_dir, exemplar_images_dir, masks_dir, background_images_dir, transform=None):
        self.source_images_dir = source_images_dir
        self.exemplar_images_dir = exemplar_images_dir
        self.masks_dir = masks_dir
        self.background_images_dir = background_images_dir
        self.transform = transform
        self.source_files = os.listdir(source_images_dir)
        self.exemplar_files = os.listdir(exemplar_images_dir)
        self.mask_files = os.listdir(masks_dir)
        self.background_files = os.listdir(background_images_dir)

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        exemplar_file = self.exemplar_files[idx]
        mask_file = self.mask_files[idx]
        background_file = self.background_files[idx]

        source_image = cv2.imread(os.path.join(self.source_images_dir, source_file), cv2.IMREAD_COLOR)
        exemplar_image = cv2.imread(os.path.join(self.exemplar_images_dir, exemplar_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        background_image = cv2.imread(os.path.join(self.background_images_dir, background_file), cv2.IMREAD_COLOR)

        if source_image is None or exemplar_image is None or mask is None or background_image is None:
            raise ValueError(f"Failed to load one of the images: {source_file}, {exemplar_file}, {mask_file}, {background_file}")

        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        exemplar_image = cv2.cvtColor(exemplar_image, cv2.COLOR_BGR2RGB)
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        mask = mask / 255.0  # Normalize mask to [0, 1]

        if self.transform:
            source_image = self.transform(source_image)
            exemplar_image = self.transform(exemplar_image)
            background_image = self.transform(background_image)
            # 对掩码单独处理，避免归一化
            mask = transforms.ToTensor()(mask)

        return source_image, exemplar_image, mask, background_image

# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据集路径
source_images_dir = "D:/datas/target_image"
exemplar_images_dir = "D:/datas/transparent_portrait"
masks_dir = "D:/datas/mask"
background_images_dir = "D:/datas/masked_background"

# 创建数据集和数据加载器
dataset = StreetSnapDataset(source_images_dir, exemplar_images_dir, masks_dir, background_images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_maps=64):
        super(DiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 强增强
def strong_augmentation(image):
    # 将批量中的每个图像单独处理
    augmented_images = []
    for img in image:
        # 将张量转换为PIL图像
        img = TF.to_pil_image(img)
        if np.random.rand() > 0.5:
            img = TF.hflip(img)
        if np.random.rand() > 0.5:
            img = TF.rotate(img, np.random.randint(-30, 30))
        if np.random.rand() > 0.5:
            img = TF.gaussian_blur(img, kernel_size=5, sigma=(0.1, 2.0))
        # 将PIL图像转换回张量
        img = TF.to_tensor(img)
        augmented_images.append(img)
    # 将列表中的张量堆叠成一个批量
    return torch.stack(augmented_images)

# 内容瓶颈
def content_bottleneck(exemplar_image):
    with torch.no_grad():
        # 将批量中的每个图像单独处理
        clip_features_list = []
        for img in exemplar_image:
            # 将张量转换为PIL图像
            img = TF.to_pil_image(img)
            # 使用 CLIP 的预处理函数
            img = preprocess(img).unsqueeze(0).to(device)
            # 提取 CLIP 特征
            clip_features = clip_model.encode_image(img)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_features_list.append(clip_features)
        # 将特征列表堆叠成一个批量
        clip_features = torch.cat(clip_features_list, dim=0)
    return clip_features

# 分类器自由引导
def classifier_free_guidance(clip_features, scale=5.0):
    unconditional_features = torch.randn_like(clip_features)
    return clip_features + scale * (clip_features - unconditional_features)

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

        # 强增强
        exemplar_image = strong_augmentation(exemplar_image)

        # 内容瓶颈
        clip_features = content_bottleneck(exemplar_image)

        # 分类器自由引导
        clip_features = classifier_free_guidance(clip_features)

        # 前向传播
        optimizer.zero_grad()
        output = model(exemplar_image * mask + background_image * (1 - mask))
        loss = criterion(output, source_image)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "D:/BaiduSyncdisk/streetphoto/Street_photography/models/diffmodel.pth")