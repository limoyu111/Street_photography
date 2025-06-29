import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import cv2


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


# --- 2. 配置输入和输出文件路径 ---
PERSON_IMAGE_PATH = "person_input.png"
BACKGROUND_IMAGE_PATH = "background_input.png"
MASK_PATH = "mask.png"
MODEL_PATH = "harmonization_model.pth"
OUTPUT_IMAGE_PATH = "harmonized_output.png"
IMAGE_SIZE = 512


def run_harmonization():
    """主函数，执行单张图片的和谐化处理"""
    # --- 检查文件是否存在 ---
    for path in [PERSON_IMAGE_PATH, BACKGROUND_IMAGE_PATH, MASK_PATH, MODEL_PATH]:
        if not os.path.exists(path):
            print(f"错误：找不到必需文件 '{path}'")
            return

    # --- 3. 加载训练好的模型 ---
    print("正在加载AI模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"模型加载成功，使用设备: {device}")

    # --- 4. 准备模型输入 ---
    print("正在准备输入图片...")
    person_bgr = cv2.imread(PERSON_IMAGE_PATH)
    background_bgr = cv2.imread(BACKGROUND_IMAGE_PATH)
    mask_gray = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    # 将所有图片尺寸调整为一致
    person_resized = cv2.resize(person_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    background_resized = cv2.resize(background_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    mask_resized = cv2.resize(mask_gray, (IMAGE_SIZE, IMAGE_SIZE))

    # 对蒙版边缘进行羽化（高斯模糊）---
    # ksize的两个值都必须是奇数，值越大，羽化/模糊效果越强。
    print("对蒙版边缘进行羽化处理以实现平滑过渡...")
    mask_blurred = cv2.GaussianBlur(mask_resized, (9, 9), 0)

    # 创建“假的”合成图
    # 使用羽化后的蒙版进行合成
    mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
    person_float = person_resized / 255.0
    background_float = background_resized / 255.0

    composite_float = background_float * (1 - mask_3channel) + person_float * mask_3channel
    composite_uint8 = (composite_float * 255).astype(np.uint8)

    composite_pil = Image.fromarray(cv2.cvtColor(composite_uint8, cv2.COLOR_BGR2RGB))
    # 输入给AI模型的蒙版仍然是清晰的原始蒙版，以确保AI知道准确的和谐化区域
    mask_pil = Image.fromarray(mask_resized)

    # --- 5. 执行AI和谐化 ---
    print("正在进行AI和谐化处理...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    composite_tensor = transform(composite_pil).unsqueeze(0).to(device)
    mask_tensor = transform(mask_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        harmonized_color = model(composite_tensor, mask_tensor)
        # 最终合成时，也使用羽化后的蒙版
        final_output_tensor = composite_tensor * (
                    1 - transform(Image.fromarray(mask_blurred)).to(device)) + harmonized_color * transform(
            Image.fromarray(mask_blurred)).to(device)

    # --- 6. 保存结果 ---
    final_output_image = transforms.ToPILImage()(final_output_tensor.squeeze(0).cpu())
    final_output_image.save(OUTPUT_IMAGE_PATH)

    print(f"\n处理完成！")
    print(f"最终合成图片已保存为: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    run_harmonization()