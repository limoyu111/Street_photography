import torch
import random
import numpy as np
import cv2
from einops import rearrange
import math
import sys
from torch import nn
from easydict import EasyDict

def generate_cross_attention_map_with_matrix(output_size, matrix, gaussian_size):
    '''
    :param output_size: (w,h)
    :param matrix: (3,3)
    :return: (h w) x (h w)
    '''
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    matrix = matrix.float()
    w,h = output_size
    ca_map  = torch.zeros(h, w, h, w)
    x_range = torch.arange(w)
    y_range = torch.arange(h)
    y, x = torch.meshgrid(y_range, x_range)
    source_coords = torch.stack([x, y], dim=2).float()
    source_coords = rearrange(source_coords, 'h w c -> (h w) c')
    ones = torch.ones_like(source_coords[:,0:1])
    source_coords_o = torch.cat([source_coords, ones], dim=1) # (h w),3
    target_coords_o = torch.matmul(matrix, source_coords_o.permute(1,0)).permute(1,0) # (h w) 3
    target_coords_o = target_coords_o / (target_coords_o[:,-1:] + 1e-9)
    target_coords_o[:,0] = torch.clip(target_coords_o[:,0], min=0, max=w)
    target_coords_o[:,1] = torch.clip(target_coords_o[:,1], min=0, max=h)
    target_coords = rearrange(target_coords_o[:,:2], '(h w) c -> h w c', h=h)
    target_coords = torch.round(target_coords).int()

    for i in range(h):
        for j in range(w):
            peak_loc = (target_coords[i,j,0], target_coords[i,j,1])
            ca_map[i,j] = generate_2dgaussian(peak_loc, (w,h), gaussian_size).permute(1,0)
    return ca_map

def generate_cross_attention_map(output_size=(16,16), gaussian_size=3):
    '''
    :param output_size: (w,h)
    :return: (h w) x (h w)
    '''
    w, h = output_size
    ca_map = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            peak_loc = (i,j)
            ca_map[i,j] = generate_2dgaussian(peak_loc, (w,h), gaussian_size)
    ca_map = rearrange(ca_map, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return ca_map

def generate_2dgaussian(peak_location, output_size, gaussian_size=3):
    '''
    :param peak_location: (x,y)
    :param output_size:   (w,h)
    :param gaussian_size: int scalar
    :return:
    '''
    gaussian_peak = get_gaussian_peak(gaussian_size * 2 + 1)  # n x n
    x,y = peak_location
    w,h = output_size
    left  = gaussian_size if x - gaussian_size >= 0 else x
    right = gaussian_size if x + gaussian_size < w  else (w-x-1)
    top   = gaussian_size if y - gaussian_size >= 0 else y
    bottom= gaussian_size if y + gaussian_size < h  else (h-y-1)
    mask  = np.zeros((w,h))
    # print(x-left, x+right+1, y-top, y+bottom+1, gaussian_size-left, gaussian_size+right+1,
    #       gaussian_size - top, gaussian_size + bottom + 1)
    mask[(x - left) : (x + right + 1), (y - top) : (y + bottom + 1)] = \
        gaussian_peak[(gaussian_size - left) : (gaussian_size + right + 1), (gaussian_size - top) : (gaussian_size + bottom + 1)]
    mask = torch.from_numpy(mask).float()
    return mask

def normalize(x):
    max = x.max()
    min = x.min()
    return (x - min) / (max - min)

def get_gauss(n):
    u = 0  # mean μ
    sig = math.sqrt(1)  # std δ
    x = np.linspace(u - 3 * sig, u + 3 * sig, n)
    y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    y = normalize(y)
    return y

def get_gaussian_peak(n):
    gauss1 = get_gauss(n).reshape(n, 1)
    gauss2 = get_gauss(n).reshape(1, n)
    # print(gauss1.shape, gauss2.shape)
    gauss_matrix = gauss1 * gauss2
    return gauss_matrix

class CrossAttentionLossV0(nn.Module):
    def __init__(self, alpha, gaussian_size):
        super().__init__()
        self.alpha = alpha
        camap = generate_cross_attention_map((16,16), gaussian_size)
        self.register_buffer('gt_camap', camap)

    def forward(self, pre_camap, indicator):
        '''
        :param pre_camap: list of predicted cross-attention map, shape [b, hw, hw]
        :param gt_camap: ground-truth cross-attention map, shape [b, hw, hw]  or [hw, hw]
        :param mask: foreground-mask, shape [b, h, w]
        :return:
        '''
        loss = 0.0
        index = (indicator[:,0] == 0)
        nonzero = torch.count_nonzero(index)
        if nonzero == 0:
            return loss
        if not isinstance(pre_camap, list):
            pre_camap = [pre_camap]
        B = pre_camap[0].shape[0]
        gt_camap = self.gt_camap.unsqueeze(0).repeat(B, 1, 1) # b,h,w
        
        for camap in pre_camap:
            per_loss = self.cross_attention_map_loss(camap, gt_camap)
            per_loss = (index * per_loss).sum() / nonzero 
            loss += per_loss
        return loss

    def cross_attention_map_loss(self, pre_camap, gt_camap):
        diff = (pre_camap - gt_camap)
        loss = torch.where(diff < 0, diff**2, self.alpha * diff**2)
        loss = loss.sum(-1).mean(-1)
        return loss

class CrossAttentionLoss(nn.Module):
    def __init__(self, alpha, gaussian_size):
        super().__init__()
        self.alpha = alpha
        camap = torch.eye(256, dtype=torch.float32)
        self.register_buffer('gt_camap', camap)

    def forward(self, pre_camap, indicator):
        '''
        :param pre_camap: list of predicted cross-attention map, shape [b, hw, hw]
        :param gt_camap: ground-truth cross-attention map, shape [b, hw, hw]  or [hw, hw]
        :param mask: foreground-mask, shape [b, h, w]
        :return:
        '''
        loss = 0.0
        index = (indicator[:,1] == 0)
        nonzero = torch.count_nonzero(index)
        if nonzero == 0:
            return loss
        if not isinstance(pre_camap, list):
            pre_camap = [pre_camap]
        B = pre_camap[0].shape[0]
        gt_camap = self.gt_camap.unsqueeze(0).repeat(B, 1, 1) # b,h,w
        
        for camap in pre_camap:
            per_loss = self.cross_attention_map_loss(camap, gt_camap)
            per_loss = (index * per_loss).sum() / nonzero 
            loss += per_loss
        return loss

    def cross_attention_map_loss(self, pre_camap, gt_camap):
        diff = (pre_camap - gt_camap)
        loss = torch.where(diff < 0, diff**2, self.alpha * diff**2)
        loss = loss.sum(-1).mean(-1)
        return loss


if __name__ == '__main__':
    bs = 5
    indicator = torch.randint(0, 2, (bs, 2))
    pre_attn = [torch.softmax(torch.randn(bs, 256, 256),dim=-1) for _ in range(4)]
    loss_func = CrossAttentionLoss(0.01, 3)
    gt_attn = loss_func.gt_camap
    attn = np.uint8(gt_attn.numpy() * 255)
    print(attn)
    cv2.imwrite('outputs/cross_attention_map.png', attn)

    loss = loss_func(pre_attn, indicator)
    print(loss)