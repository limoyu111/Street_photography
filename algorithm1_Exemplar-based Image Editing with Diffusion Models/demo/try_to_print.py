import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义扩散模型
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

# 加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel().to(device)
model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path, transform):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    return image

# 加载背景图像、示例图像和掩码
background_image_path = "C:/Users/36298/Desktop/input/back (1).png"
exemplar_image_path = "C:/Users/36298/Desktop/datas/transparent_portrait/jp (596).png"
mask_path = "C:/Users/36298/Desktop/datas/mask/jp (596).png"
#再这里协商你要输入的图片的地址
background_image = preprocess_image(background_image_path, transform).unsqueeze(0).to(device)
exemplar_image = preprocess_image(exemplar_image_path, transform).unsqueeze(0).to(device)
mask = preprocess_image(mask_path, transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])).unsqueeze(0).to(device)

# 生成合成图像
def generate_image(model, background_image, exemplar_image, mask, device):
    model.eval()
    with torch.no_grad():
        background_image = background_image.to(device).float()
        exemplar_image = exemplar_image.to(device).float()
        mask = mask.to(device).float()
        output = model(exemplar_image * mask + background_image * (1 - mask))
        return output

synthesized_image = generate_image(model, background_image, exemplar_image, mask, device)

# 保存和显示合成图像
output_image_path = "C:/Users/36298/Desktop/output/create.png"
torchvision.utils.save_image(synthesized_image, output_image_path, normalize=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(background_image[0].permute(1, 2, 0).cpu().numpy())
plt.title("Background Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(exemplar_image[0].permute(1, 2, 0).cpu().numpy())
plt.title("Exemplar Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(synthesized_image[0].permute(1, 2, 0).cpu().numpy())
plt.title("Synthesized Image")
plt.axis("off")

plt.show()