import os
import cv2
import numpy as np
import torch
from torchvision import models
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS
from scipy import linalg

# 定义计算Inception Score的函数
def calculate_inception_score(images, model, batch_size=32, splits=10):
    model.eval()
    scores = []
    for i in range(splits):
        subset = images[i * batch_size:(i + 1) * batch_size]
        subset = torch.cat(subset, dim=0)
        preds = model(subset)[0]
        preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        kl_divs = []
        for j in range(batch_size):
            split = preds[j:j + 1]
            kl_div = split * (np.log(split) - np.log(np.mean(preds, axis=0)))
            kl_divs.append(np.sum(kl_div))
        scores.append(np.exp(np.mean(kl_divs)))
    return np.mean(scores), np.std(scores)

# 定义计算Frechet Inception Distance的函数
def calculate_fid(images, model, batch_size=32):
    model.eval()
    act = []
    for i in range(0, len(images), batch_size):
        subset = images[i:i + batch_size]
        subset = torch.cat(subset, dim=0)
        pred = model(subset)[0]
        pred = pred.view(pred.size(0), -1)
        act.append(pred.cpu().numpy())
    act = np.concatenate(act, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

# 定义加载图像的函数
def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = ToTensor()(image)
                images.append(image.unsqueeze(0))
    return images

# 定义评估相似度的函数
def evaluate_similarity(generated_images, target_images):
    mse_values = []
    ssim_values = []
    psnr_values = []
    lpips_values = []

    lpips_model = LPIPS(net='vgg').eval()

    for generated_image, target_image in zip(generated_images, target_images):
        # Convert PyTorch tensors to NumPy arrays
        generated_image_np = generated_image.squeeze(0).permute(1, 2, 0).numpy() * 255
        target_image_np = target_image.squeeze(0).permute(1, 2, 0).numpy() * 255

        generated_gray = cv2.cvtColor(generated_image_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_image_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        mse = np.mean((generated_gray - target_gray) ** 2)
        ssim_value = ssim(generated_gray, target_gray, data_range=255)
        psnr_value = psnr(target_gray, generated_gray, data_range=255)

        generated_tensor = ToTensor()(generated_image_np.astype(np.uint8)).unsqueeze(0)
        target_tensor = ToTensor()(target_image_np.astype(np.uint8)).unsqueeze(0)
        lpips_value = lpips_model(generated_tensor, target_tensor).item()

        mse_values.append(mse)
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        lpips_values.append(lpips_value)

        print(f"MSE: {mse:.4f}, SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}, LPIPS: {lpips_value:.4f}")

    print(f"Average MSE: {np.mean(mse_values):.4f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
    print(f"Average PSNR: {np.mean(psnr_values):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_values):.4f}")

# 主函数
def main():
    generated_folder = "C:/Users/36298/Desktop/testdatas/out"
    target_folder = "C:/Users/36298/Desktop/testdatas/target"

    generated_images = load_images_from_folder(generated_folder)
    target_images = load_images_from_folder(target_folder)

    if len(generated_images) == 0 or len(target_images) == 0:
        print("No images found in one or both folders. Exiting.")
        return

    evaluate_similarity(generated_images, target_images)

    # 加载Inception模型
    inception_model = models.inception_v3(pretrained=True, aux_logits=False)

    # 计算Inception Score
    is_mean, is_std = calculate_inception_score(generated_images, inception_model)
    print(f"Inception Score: Mean={is_mean:.4f}, Std={is_std:.4f}")

    # 计算Frechet Inception Distance
    mu1, sigma1 = calculate_fid(generated_images, inception_model)
    mu2, sigma2 = calculate_fid(target_images, inception_model)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Frechet Inception Distance: {fid:.4f}")

if __name__ == "__main__":
    main()