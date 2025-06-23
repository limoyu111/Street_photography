from re import X
import torch
from kornia.filters import gaussian_blur2d
import sys, os
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import minimize
import scipy.optimize as spo
import torch.nn.functional as F
import copy

def fill_mask(mask, bbox, kernel_size, sigma):
    x1, y1, x2, y2 = bbox
    out_mask = copy.deepcopy(mask)
    local_mask = out_mask[:, :, y1:y2, x1:x2]
    print('local', local_mask.shape, kernel_size, sigma)
    local_mask = gaussian_blur2d(local_mask, kernel_size, sigma) # border_type='constant'
    local_mask = torch.where(local_mask > 1e-5, 1., 0.).float()
    out_mask[:, :, y1:y2, x1:x2] = local_mask
    return out_mask

def bbox2mask(bbox, mask_w, mask_h): 
    mask = torch.zeros((1, 1, mask_h, mask_w), dtype=torch.float32)
    mask[:, :, bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1.
    return mask

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def get_kernel_size(sigma):
    if isinstance(sigma, (float, int)):
        kernel_size = int(3 * sigma)
        if kernel_size & 1 == 0:
            kernel_size += 1
        return (kernel_size, kernel_size)
    else:
        sigma_h, sigma_w = sigma
        kernel_w = int(sigma_w * 3) if int(sigma_w * 3) & 1 == 1 else int(sigma_w * 3 - 1)
        kernel_w = max(kernel_w, 1)
        kernel_h = int(sigma_h * 3) if int(sigma_h * 3) & 1 == 1 else int(sigma_h * 3 - 1)
        kernel_h = max(kernel_h, 1)
        kernel_size = (kernel_h, kernel_w)
        return kernel_size
    
import cv2
def reverse_mask_tensor(tensor):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = (512, 512)):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def compute_mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    ) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """
    if mask1.ndim == 4:
        mask1.squeeze_(1)
    if mask2.ndim == 4:
        mask2.squeeze_(1)
    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,
    )
    return ret.mean()
    
cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
configs  = OmegaConf.load(cfg_path).data.params.train
dataset  = instantiate_from_config(configs)
dataloader = DataLoader(dataset=dataset, 
                        batch_size=4, 
                        shuffle=False,
                        num_workers=8)
print('{} samples = {} bs x {} batches'.format(
    len(dataset), dataloader.batch_size, len(dataloader)
))
vis_dir = os.path.join(proj_dir, 'outputs/test_dataaug/batch_data')
os.makedirs(vis_dir, exist_ok=True)

for _,batch in enumerate(dataloader):
    gtmask_t  = batch['gt_mask']
    bgmask_t  = 1 - batch['bg_mask']
    gtbbox_norm = batch['gt_bbox']
    gtbbox_int  = (gtbbox_norm * gtmask_t.shape[-1]).int()
    bbox_norm = batch["bbox"]
    bbox_int  = (bbox_norm * gtmask_t.shape[-1]).int()
    for i in range(gtmask_t.shape[0]):
        img_w,  img_h  = gtmask_t.shape[-2], gtmask_t.shape[-1]
        bbox_w, bbox_h = (bbox_int[i][2]-bbox_int[i][0]).item(), (bbox_int[i][3]-bbox_int[i][1]).item()
        # sigma = (bbox_h * 2. / 3., bbox_w * 2. / 3.)
        # kernel_size = get_kernel_size(sigma)
        # print(sigma, kernel_size, bbox_w, bbox_h)
        kernel_size = (int(bbox_h * 2 - 1), int(bbox_w * 2 - 1))
        sigma = (kernel_size[0] / 3., kernel_size[1] / 3.)
        blur_mask = fill_mask(gtmask_t[i:i+1], bbox_int[i], kernel_size, sigma)
        iou = compute_mask_iou(blur_mask, bgmask_t[i:i+1])
        # blur_mask = reverse_mask_tensor(blur_mask)
        # bg_mask = reverse_mask_tensor(bgmask_t[j:j+1])
        print('iou', iou)