from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import reverse
from cmath import inf
from curses.panel import bottom_panel
from dis import dis
from email.mime import image

import os
from io import BytesIO
import json
import logging
import base64
from sre_parse import State
from sys import prefix
import threading
import random
from turtle import left, right
from cv2 import norm
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
from sympy import source
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
from tqdm import tqdm
import sys

import transformers
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True, resize=True, image_size=(512, 512)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True, resize=True, image_size=(224, 224)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def scan_all_files():
    bbox_dir = os.path.join(proj_dir, '../../dataset/open-images/bbox_mask')
    assert os.path.exists(bbox_dir), bbox_dir
    
    bad_files = []
    for split in os.listdir(bbox_dir):
        total_images, total_pairs, bad_masks, bad_images = 0, 0, 0, 0
        subdir = os.path.join(bbox_dir, split)
        if not os.path.isdir(subdir) or split not in ['train', 'test', 'validation']:
            continue
        for file in tqdm(os.listdir(subdir)):
            try:
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        info = line.split(' ')
                        mask_file = os.path.join(bbox_dir, '../masks', split, info[-2])
                        if os.path.exists(mask_file):
                            total_pairs += 1
                        else:
                            bad_masks += 1
                total_images += 1
            except:
                bad_files.append(file)
                bad_images += 1
        print('{}, {} images({} bad), {} pairs({} bad)'.format(
            split, total_images, bad_images, total_pairs, bad_masks))
        
        if len(bad_files) > 0:
            with open(os.path.join(bbox_dir, 'bad_files.txt'), 'w') as f:
                for file in bad_files:
                    f.write(file + '\n')
        
    print(f'{len(bad_files)} bad_files')
    
class DataAugmentation:
    def __init__(self, 
                 fg_augtype='AG', 
                 appearance_prob=0.5,
                 geometric_prob=0.5,
                 fg_augregion='global', 
                 bg_augtype='crop',
                 mask_augtype='pad'):
        assert fg_augtype in ['A', 'G', 'AG', 'none'] and fg_augregion in ['global', 'local'], (fg_augtype, fg_augregion)
        assert bg_augtype in ['crop', 'none'], bg_augtype
        assert mask_augtype in ['pad', 'none'], mask_augtype
        self.fg_augtype   = fg_augtype
        self.fg_augregion = fg_augregion
        self.mask_augtype = mask_augtype
        self.appearance_trans = A.Compose([
            A.Blur(p=0.3),
            A.ColorJitter(brightness=0.5, contrast=0.5, 
                          saturation=0.5, hue=0.05, 
                          always_apply=False, p=1)
        ])
        self.appearance_prob = appearance_prob
        border_mode = 0
        self.geometric_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20,
                     border_mode=border_mode,
                     value=(127,127,127),
                     mask_value=0,
                     p=0.5),
            A.Perspective(scale=(0.05, 0.1), 
                          pad_mode=border_mode,
                          pad_val =(127,127,127),
                          mask_pad_val=0,
                          fit_output=True, 
                          p=1)
        ])
        self.geometric_prob = geometric_prob
        self.cropbg_p  = 0.5
        self.padmask_p = 0.5
        self.bbox_maxlen = 0.8
        
    def augment_foreground(self, image, mask):
        origin_img  = copy.deepcopy(image)
        origin_mask = copy.deepcopy(mask) 
        if 'A' in self.fg_augtype and np.random.rand() < self.appearance_prob:
            transformed = self.appearance_trans(image=image)
            image = transformed['image']
        if 'G' in self.fg_augtype and np.random.rand() < self.geometric_prob:
            transformed = self.geometric_trans(image=image, mask=mask)
            image = transformed['image']
            mask  = transformed['mask']
        if self.fg_augregion == 'local':
            # only change the object and remain the other region
            image_f = origin_img.astype(np.float32)
            gray_image = np.ones_like(image_f) * 127
            mask_f = origin_mask[:,:,np.newaxis].astype(np.float32) / 255
            trans_mask_f = mask[:,:,np.newaxis].astype(np.float32) / 255
            trans_img_f  = image.astype(np.float32)
            # mask the original foreground region
            image_f = image_f * (1 - mask_f) + gray_image * mask_f
            # copy-paste the transformed foreground
            image_f = image_f * (1 - trans_mask_f) + trans_mask_f * trans_img_f
            image   = np.uint8(image_f) 
        return image, mask
        
    def __call__(self, bg_img, bbox, bg_mask, fg_img, fg_mask):
        fg_img, fg_mask = self.augment_foreground(fg_img, fg_mask)
        bg_img, bbox, bg_mask = self.augment_background(bg_img, bbox, bg_mask)
        pad_bbox, pad_mask = self.augment_mask(bbox, bg_img)
        return {
            "fg_image": fg_img,
            "fg_mask":  fg_mask,
            "bg_image": bg_img,
            "bg_mask":  bg_mask,
            "bbox":     bbox,
            "pad_bbox": pad_bbox,
            "pad_mask": pad_mask
        }
    
    def augment_mask(self, bbox, bg_img):
        pad_bbox = self.random_pad_bbox(bbox, bg_img.shape[1], bg_img.shape[0])
        pad_mask = bbox2mask(pad_bbox, bg_img.shape[1], bg_img.shape[0])
        return pad_bbox, pad_mask

    
    def augment_background(self, image, bbox, mask):
        trans_bbox  = copy.copy(bbox)
        trans_image = image.copy()
        trans_mask  = mask.copy() 
        if np.random.rand() < self.cropbg_p:
            width, height = image.shape[1], image.shape[0]
            bbox_w = float(bbox[2] - bbox[0]) / width
            bbox_h = float(bbox[3] - bbox[1]) / height
            
            left, right, top, down = 0, width, 0, height 
            if bbox_w < self.bbox_maxlen:
                maxcrop = (width - bbox_w * width / self.bbox_maxlen) / 2
                left  = int(np.random.rand() * min(maxcrop, bbox[0]))
                right = width - int(np.random.rand() * min(maxcrop, width - bbox[2]))

            if bbox_h < self.bbox_maxlen:
                maxcrop = (height - bbox_h * height / self.bbox_maxlen) / 2
                top   = int(np.random.rand() * min(maxcrop, bbox[1]))
                down  = height - int(np.random.rand() * min(maxcrop, height - bbox[3]))
            
            trans_bbox = [bbox[0] - left, bbox[1] - top, bbox[2] - left, bbox[3] - top]
            trans_image = trans_image[top:down, left:right]
            trans_mask  = trans_mask[top:down, left:right]
            # print(image.shape, trans_image.shape, trans_mask.shape, bbox, trans_bbox)
        return trans_image, trans_bbox, trans_mask
    
    def random_pad_bbox(self, bbox, width, height):
        bbox_pad  = copy.copy(bbox)
        if np.random.rand() < self.padmask_p:
            bbox_w = float(bbox[2] - bbox[0]) / width
            bbox_h = float(bbox[3] - bbox[1]) / height
            
            if bbox_w < self.bbox_maxlen:
                maxpad = width * (self.bbox_maxlen - bbox_w) / 2
                bbox_pad[0] = max(0, int(bbox[0] - np.random.rand() * maxpad))
                bbox_pad[2] = min(width, int(bbox[2] + np.random.rand() * maxpad))
            
            if bbox_h < self.bbox_maxlen:
                maxpad = height * (self.bbox_maxlen - bbox_h) / 2
                bbox_pad[1] = max(0, int(bbox[1] - np.random.rand() * maxpad))
                bbox_pad[3] = min(height, int(bbox[3] + np.random.rand() * maxpad))
        return bbox_pad
    
def bbox2mask(bbox, mask_w, mask_h):
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
        return mask
    
def constant_pad_bbox(bbox, width, height, value=10):
    ### Get reference image
    bbox_pad=copy.copy(bbox)
    left_space  = bbox[0]
    up_space    = bbox[1]
    right_space = width  - bbox[2]
    down_space  = height - bbox[3] 

    bbox_pad[0]=bbox[0]-min(value, left_space)
    bbox_pad[1]=bbox[1]-min(value, up_space)
    bbox_pad[2]=bbox[2]+min(value, right_space)
    bbox_pad[3]=bbox[3]+min(value, down_space)
    return bbox_pad
    
    
def crop_image_by_bbox(img, bbox):
    if isinstance(img, np.ndarray):
        width,height = img.shape[1], img.shape[0]
    else:
       width,height = img[0].shape[1], img[0].shape[0]
    bbox_pad = constant_pad_bbox(bbox, width, height, 10)
    
    if isinstance(img, (list, tuple)):
        crop = [per_img[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]].copy() for per_img in img]
    else:
        crop = img[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]].copy()
    return crop

def image2inpaint(image, mask):
    if len(mask.shape) == 2:
        mask_f = mask[:,:,np.newaxis].copy()
    else:
        mask_f = mask.copy()
    mask_f  = mask_f.astype(np.float32) / 255
    inpaint = image.astype(np.float32)
    gray  = np.ones_like(inpaint) * 127
    inpaint = inpaint * (1 - mask_f) + mask_f * gray
    inpaint = np.uint8(inpaint)
    return inpaint
    
def test_data_augmentation():
    split = 'train'
    dataset_dir = os.path.join(proj_dir, '../../dataset/open-images')
    bbox_dir    = os.path.join(dataset_dir, 'bbox_mask', split)
    image_dir   = os.path.join(dataset_dir, 'images', split)
    mask_dir    = os.path.join(dataset_dir, 'masks', split) 
    assert os.path.exists(bbox_dir), bbox_dir
    
    random_trans = DataAugmentation()
    vis_dir = os.path.join(proj_dir, 'outputs/test_dataaug/border_replicate')
    os.makedirs(vis_dir, exist_ok=True)
    
    def mask2bgr(mask, image_size=(256, 256)):
        return cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), image_size)
    
    def image2bgr(image, image_size=(256, 256)):
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_size)
    count = 0
    for file in os.listdir(bbox_dir):
        with open(os.path.join(bbox_dir, file), 'r') as f:
            im_name  = file.split('.')[0]
            img_path = os.path.join(image_dir, im_name + '.jpg')
            # img_p_np = cv2.imread(img_path)
            # img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
            img_p_np = np.asarray(Image.open(img_path))
            
            for line in f.readlines():
                info  = line.strip().split(' ')
                label = info[-3]
                conf  = float(info[-1])
                bbox  = [int(float(f)) for f in info[:4]]
                mask_path = os.path.join(mask_dir, info[-2])
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (img_p_np.shape[1], img_p_np.shape[0]))
                else:
                    continue
                fg_img, fg_mask  = crop_image_by_bbox([img_p_np, mask], bbox)
                image_size = (256, 256)
                bbox_mask = bbox2mask(bbox, mask.shape[1], mask.shape[0])
                ver_border = np.ones((image_size[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
                src  = image2bgr(img_p_np)
                src_mask = mask2bgr(mask)
                src_inpaint = image2bgr(image2inpaint(img_p_np, bbox_mask))
                src_fg   = image2bgr(fg_img)
                src_fgmask = mask2bgr(fg_mask)
                cat_img = np.concatenate([src, ver_border, src_mask, ver_border, src_inpaint, ver_border, src_fg, ver_border, src_fgmask], axis=1)
                img_list = [cat_img]
                for i in range(3):
                    transformed = random_trans(img_p_np, bbox, mask, fg_img, fg_mask)
                    inpaint = image2bgr(image2inpaint(transformed['bg_image'], transformed['pad_mask']))
                    gt_mask = mask2bgr(transformed['bg_mask'])
                    text    = 'confidence:{:.2f}'.format(conf)
                    cv2.putText(gt_mask, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cat_img = np.concatenate([
                        image2bgr(transformed['bg_image']), ver_border, gt_mask, ver_border, inpaint,
                        ver_border, image2bgr(transformed['fg_image']), ver_border, mask2bgr(transformed['fg_mask'])
                    ], axis=1)
                    hor_border = np.ones((10, img_list[-1].shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
                    img_list.append(hor_border)
                    img_list.append(cat_img)
                cat_img = np.concatenate(img_list, axis=0)
                cv2.imwrite(os.path.join(vis_dir, f'{im_name}_{label}_.jpg'), cat_img)
        count += 1
        if count > 20:
            exit()

def check_dir(dir):
    assert os.path.exists(dir), dir
    return dir

def get_bbox_tensor(bbox, width, height):
    norm_bbox = copy.deepcopy(bbox)
    norm_bbox = torch.tensor(norm_bbox).reshape(-1).float()
    norm_bbox[0::2] /= width
    norm_bbox[1::2] /= height
    return norm_bbox
    
def reverse_image_tensor(tensor):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor.float() + 1) / 2
    tensor = torch.permute(tensor, (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = (512, 512)):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

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

def reverse_clip_tensor(tensor):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073],  dtype=torch.float)
    MEAN = MEAN.reshape(1, 3, 1, 1).to(tensor.device)
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float)
    STD  = STD.reshape(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor * STD) + MEAN
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = (512, 512)):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list
    

class OpenImageDataset(data.Dataset):
    def __init__(self,split,**args):
        self.split=split
        dataset_dir = args['dataset_dir']
        self.bbox_dir = check_dir(os.path.join(dataset_dir, 'bbox_mask', split))
        self.image_dir= check_dir(os.path.join(dataset_dir, 'images', split))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'masks', split))
        self.bbox_path_list= os.listdir(self.bbox_dir)
        self.bbox_path_list.sort()
        self.length=len(self.bbox_path_list)
        self.random_trans = DataAugmentation()
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(224, 224))
        
    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file) as f:
            for line in f.readlines():
                info  = line.strip().split(' ')
                label = info[-3]
                confidence = float(info[-1])
                bbox  = [int(float(f)) for f in info[:4]]
                mask  = os.path.join(self.mask_dir, info[-2])
                if os.path.exists(mask):
                    bbox_list.append((bbox, label, mask, confidence))
        return bbox_list
    
    def __getitem__(self, index):
        # try:
        # get bbox and mask
        bbox_file = self.bbox_path_list[index] 
        bbox_path = os.path.join(self.bbox_dir, bbox_file)
        bbox_list  = self.load_bbox_file(bbox_path)
        bbox,label,mask_path, mask_conf = random.choice(bbox_list)
        # get source image and mask
        image_path = os.path.join(self.image_dir, os.path.splitext(bbox_file)[0] + '.jpg')
        source_img = Image.open(image_path).convert("RGB")
        source_np  = np.asarray(source_img)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (source_np.shape[1], source_np.shape[0]))
        fg_img, fg_mask  = crop_image_by_bbox([source_np, mask], bbox)
        bbox_mask = bbox2mask(bbox, mask.shape[1], mask.shape[0])
        mask = np.where(bbox_mask > 127, mask, bbox_mask) 
        # perform data augmentation
        transformed = self.random_trans(source_np, bbox, mask, fg_img, fg_mask)
        img_width, img_height = transformed["bg_mask"].shape[1], transformed["bg_mask"].shape[0]
        gt_mask_tensor = self.mask_transform(Image.fromarray(transformed["bg_mask"]))
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0)
        gt_img_tensor  = self.sd_transform(Image.fromarray(transformed['bg_image']))
        fg_mask_tensor = self.clip_mask_transform(Image.fromarray(transformed['fg_mask']))
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)
        fg_img_tensor  = self.clip_transform(Image.fromarray(transformed['fg_image']))
        mask_tensor = self.mask_transform(Image.fromarray(transformed['pad_mask']))
        mask_tensor = torch.where(mask_tensor > 0, 1, 0)
        bbox_tensor = get_bbox_tensor(transformed['pad_bbox'], img_width, img_height)
        
        inpaint_tensor  = gt_img_tensor * (1 - mask_tensor) 
        return {"image_path": image_path,
                "gt_img": gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "mask_conf": mask_conf,
                "bg_img":  inpaint_tensor,
                "bg_mask": mask_tensor,
                "fg_img": fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "bbox": bbox_tensor}
        # except:
        #     idx = np.random.randint(0, len(self)-1)
        #     return self[idx]
            
    def __len__(self):
        return self.length
    
def vis_batch_data(batch):
    file = batch['image_path']
    gt_t = batch['gt_img']
    gtmask_t = batch['gt_mask']
    bg_t = batch['bg_img']
    bgmask_t  = batch['bg_mask']
    fg_t = batch['fg_img']
    fgmask_t = batch['fg_mask']
    
    gt_imgs  = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t) 
    bg_imgs  = reverse_image_tensor(bg_t)
    bg_masks = reverse_mask_tensor(bgmask_t)
    fg_imgs  = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)
    
    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    img_list = []
    for i in range(len(file)):
        im_name = os.path.basename(file[i])
        cat_img = np.concatenate([gt_imgs[i], ver_border, gt_masks[i], ver_border, bg_imgs[i], 
                                  ver_border, fg_imgs[i], ver_border, fg_masks[i]], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch
    
        
if __name__ == '__main__':
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    dataset  = instantiate_from_config(configs)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=8, 
                            shuffle=False,
                            num_workers=8)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/test_dataaug/batch_data')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        file = batch['image_path']
        gt_t = batch['gt_img']
        gtmask_t = batch['gt_mask']
        bg_t = batch['bg_img']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        fgmask_t = batch['fg_mask']

        batch_img = vis_batch_data(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break
    
    
    


