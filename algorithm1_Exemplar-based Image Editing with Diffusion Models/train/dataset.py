import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms

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