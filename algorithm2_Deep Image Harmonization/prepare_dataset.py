import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 配置原始数据路径 ---
PERSON_DIR = "./datasets/person_image/"
BACKGROUND_DIR = "./datasets/background_image/"
ORIGINAL_DIR = "./datasets/original_image/"
MASK_DIR = "./datasets/mask/"

# --- 配置处理后数据的输出路径 ---
# 脚本会自动创建这些文件夹
OUTPUT_COMPOSITE_DIR = "./train_data/composite/"
OUTPUT_REAL_DIR = "./train_data/real/"
OUTPUT_MASK_DIR = "./train_data/mask/"


def prepare():
    print("开始创建训练所需的数据集...")
    os.makedirs(OUTPUT_COMPOSITE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REAL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    image_names = sorted(os.listdir(PERSON_DIR))

    for name in tqdm(image_names, desc="Processing images"):
        person_path = os.path.join(PERSON_DIR, name)
        background_path = os.path.join(BACKGROUND_DIR, name)
        original_path = os.path.join(ORIGINAL_DIR, name)
        mask_path = os.path.join(MASK_DIR, name)

        person = cv2.imread(person_path)
        background = cv2.imread(background_path)
        original = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if person is None or background is None or original is None or mask is None:
            print(f"Skipping {name} due to loading error.")
            continue

        # 创建“假的”合成图：直接将人像按遮罩粘贴到背景上
        # 这里用背景图乘以（1-mask），再加上人像图乘以mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        composite = background * (1 - mask_3channel) + person * mask_3channel
        composite = composite.astype(np.uint8)

        # 保存处理好的图片
        cv2.imwrite(os.path.join(OUTPUT_COMPOSITE_DIR, name), composite)
        cv2.imwrite(os.path.join(OUTPUT_REAL_DIR, name), original)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, name), mask)

    print("数据集准备完成！所有文件已保存在 'train_data' 文件夹中。")


if __name__ == "__main__":
    prepare()