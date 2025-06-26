import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 支持中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
matplotlib.rcParams['font.size'] = 12  # 设置字体大小
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义扩散模型
class DiffusionModel(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_maps=64):
        super(DiffusionModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, feature_maps, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(feature_maps, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 加载训练好的模型
def load_model(model_path, device):
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 处理输入图像
def process_input_image(image_path, transform):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    return image.unsqueeze(0).to(device)

# 生成输出图像
def generate_output(model, background_image, person_image, mask):
    background_image = background_image.to(device).float()
    person_image = person_image.to(device).float()
    mask = mask.to(device).float()

    with torch.no_grad():
        output = model(person_image * mask + background_image * (1 - mask))
    return output

# 保存生成的图像
def save_image(tensor, output_path):
    image = tensor.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Generated image saved to {output_path}")

# 展示图像
def display_image(image_path, label):
    image = Image.open(image_path)
    image.thumbnail((128, 128))  # 调整图片大小以便显示
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo  # 保持对PhotoImage的引用

# GUI 应用程序
class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像生成演示")

        # 设置默认模型路径
        self.model_path = tk.StringVar(value="D:\\streetphoto\\Street_photography\\algorithm1_Exemplar-based Image Editing with Diffusion Models\\models\\diffmodel2.pth")
        self.background_image_path = tk.StringVar()
        self.person_image_path = tk.StringVar()
        self.mask_image_path = tk.StringVar()
        self.output_directory = tk.StringVar()

        self.model = None

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="模型路径：").grid(row=0, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="浏览", command=self.browse_model).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.root, text="背景图像：").grid(row=1, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.background_image_path, width=50).grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.root, text="浏览", command=self.browse_background_image).grid(row=1, column=2, padx=10, pady=10)

        tk.Label(self.root, text="人像图像：").grid(row=2, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.person_image_path, width=50).grid(row=2, column=1, padx=10, pady=10)
        tk.Button(self.root, text="浏览", command=self.browse_person_image).grid(row=2, column=2, padx=10, pady=10)

        tk.Label(self.root, text="掩码图像：").grid(row=3, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.mask_image_path, width=50).grid(row=3, column=1, padx=10, pady=10)
        tk.Button(self.root, text="浏览", command=self.browse_mask_image).grid(row=3, column=2, padx=10, pady=10)

        tk.Label(self.root, text="输出目录：").grid(row=4, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.output_directory, width=50).grid(row=4, column=1, padx=10, pady=10)
        tk.Button(self.root, text="浏览", command=self.browse_output_directory).grid(row=4, column=2, padx=10, pady=10)

        tk.Button(self.root, text="生成图像", command=self.generate_image).grid(row=5, column=1, padx=10, pady=10)

        # 显示图片的区域
        self.background_image_label = tk.Label(self.root)
        self.background_image_label.grid(row=1, column=3, padx=10, pady=10)

        self.person_image_label = tk.Label(self.root)
        self.person_image_label.grid(row=2, column=3, padx=10, pady=10)

        self.mask_image_label = tk.Label(self.root)
        self.mask_image_label.grid(row=3, column=3, padx=10, pady=10)

        self.output_image_label = tk.Label(self.root)
        self.output_image_label.grid(row=5, column=3, padx=10, pady=10)

    def browse_model(self):
        path = filedialog.askopenfilename(title="选择模型文件", filetypes=[("模型文件", "*.pth")])
        if path:
            self.model_path.set(path)

    def browse_background_image(self):
        path = filedialog.askopenfilename(title="选择背景图像", filetypes=[("图像文件", "*.jpg;*.png")])
        if path:
            self.background_image_path.set(path)
            self.display_image(path, self.background_image_label)

    def browse_person_image(self):
        path = filedialog.askopenfilename(title="选择人像图像", filetypes=[("图像文件", "*.jpg;*.png")])
        if path:
            self.person_image_path.set(path)
            self.display_image(path, self.person_image_label)

    def browse_mask_image(self):
        path = filedialog.askopenfilename(title="选择掩码图像", filetypes=[("图像文件", "*.jpg;*.png")])
        if path:
            self.mask_image_path.set(path)
            self.display_image(path, self.mask_image_label)

    def browse_output_directory(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_directory.set(path)

    def display_image(self, image_path, label):
        image = Image.open(image_path)
        image.thumbnail((128, 128))  # 调整图片大小以便显示
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo  # 保持对PhotoImage的引用

    def generate_image(self):
        try:
            if not self.model:
                self.model = load_model(self.model_path.get(), device)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),  # 调整到 512x512
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            background_image = process_input_image(self.background_image_path.get(), transform)
            person_image = process_input_image(self.person_image_path.get(), transform)

            mask_image = cv2.imread(self.mask_image_path.get(), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise ValueError(f"Failed to load mask image: {self.mask_image_path.get()}")
            mask_image = cv2.resize(mask_image, (512, 512))  # 调整到 512x512
            mask_image = torch.tensor(mask_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

            output_image = generate_output(self.model, background_image, person_image, mask_image)

            output_directory = self.output_directory.get()
            os.makedirs(output_directory, exist_ok=True)
            output_image_path = os.path.join(output_directory, "output.jpg")
            save_image(output_image, output_image_path)

            # 展示生成的图像
            self.display_image(output_image_path, self.output_image_label)

            messagebox.showinfo("成功", "图像生成成功！")
        except Exception as e:
            messagebox.showerror("错误", str(e))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()