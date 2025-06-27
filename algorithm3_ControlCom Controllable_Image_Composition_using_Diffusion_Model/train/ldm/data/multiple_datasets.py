import os
import random
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image
import torch.utils.data as data
from torch.utils.data.dataset import ConcatDataset
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
from tqdm import tqdm
import sys
import shutil
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)
from ldm.data.open_images_control import *

class MVImgNetDataset(data.Dataset):
    def __init__(self, split, **args):
        self.split  = split
        self.dataset_dir = 'path-to-MVImgNet'
        # assert os.path.exists(self.dataset_dir), self.dataset_dir
        self.parse_augment_config(args)
        self.sample_dict = self.generate_sample_list('path-to-data-dir/MVImgNet_sample_tree.json')
        self.sample_list = [k for k in self.sample_dict.keys() if 'image' in self.sample_dict[k]]
        if self.split == 'debug':
            self.sample_list = self.sample_list[:100] 
        self.length = len(self.sample_list)

        self.random_trans = MVDataAugmentation(self.augment_background, self.augment_box)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(16, 16))

    def parse_augment_config(self, args):
        self.augment_config = args['augment_config'] if 'augment_config' in args else None
        if self.augment_config:
            self.sample_mode = self.augment_config.sample_mode
            self.augment_types = self.augment_config.augment_types
            self.sample_prob = self.augment_config.sample_prob
            if self.sample_mode == 'random':
                assert len(self.augment_types) == len(self.sample_prob), \
                    'len({}) != len({})'.format(self.augment_types, self.sample_prob)
            self.augment_background = self.augment_config.augment_background
            self.augment_box = self.augment_config.augment_box
            self.replace_background_prob = self.augment_config.replace_background_prob
            self.use_inpaint_background  = self.augment_config.use_inpaint_background
        else:
            self.sample_mode   = 'random'
            self.augment_types = [(0,0), (0,1), (1,0), (1,1)]
            self.sample_prob = [1. / len(self.augment_types)] * len(self.augment_types)
            self.augment_background = False
            self.augment_box = False
            self.replace_background_prob = 1
            self.use_inpaint_background  = True
        self.augment_list = list(range(len(self.augment_types)))

    def generate_sample_list(self, tree_path):
        file_tree = json.load(open(tree_path, 'r'))
        return file_tree

    def load_bbox_file(self, bbox_file):
        bboxes = read_pcache_txt(bbox_file)
        bboxes = bboxes[0].split()
        bbox = list(map(int, bboxes[:4]))
        return bbox
    
    def sample_all_augmentations(self, source_np, bbox, mask, 
                                 fg_img, fg_mask, new_bg,
                                 src_fgimg, src_fgmask, src_newbg):
        output = {}
        for indicator in self.augment_types:
            sample = self.sample_one_augmentations(source_np, bbox, mask, 
                                                   fg_img, fg_mask, indicator, new_bg,
                                                   src_fgimg, src_fgmask, src_newbg)
            for k,v in sample.items():
                if k not in output:
                    output[k] = [v]
                else:
                    output[k].append(v)
            sample = None
        for k in output.keys():
            output[k] = torch.stack(output[k], dim=0)
        return output
    
    def sample_one_augmentations(self, source_np, bbox, mask, 
                                 fg_img, fg_mask, indicator, new_bg,
                                 src_fgimg, src_fgmask, src_newbg):
        transformed = self.random_trans(source_np, bbox, mask, 
                                        fg_img, fg_mask, indicator, new_bg,
                                        src_fgimg, src_fgmask, src_newbg)
        # get ground-truth composite image and bbox
        gt_mask = Image.fromarray(transformed["bg_mask"])
        img_width, img_height = gt_mask.size
        gt_mask_tensor = self.mask_transform(gt_mask)
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0).float() 
        
        gt_img_tensor  = Image.fromarray(transformed['bg_img'])
        gt_img_tensor  = self.sd_transform(gt_img_tensor)
        gt_bbox_tensor = transformed['bbox']
        gt_bbox_tensor = get_bbox_tensor(gt_bbox_tensor, img_width, img_height)
        mask_tensor = Image.fromarray(transformed['pad_mask'])
        mask_tensor = self.mask_transform(mask_tensor)
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        bbox_tensor = transformed['pad_bbox']
        bbox_tensor = get_bbox_tensor(bbox_tensor, img_width, img_height)
        # get foreground and foreground mask
        fg_mask_tensor = Image.fromarray(transformed['fg_mask'])
        fg_mask_tensor = self.clip_mask_transform(fg_mask_tensor)
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)
        gt_fg_mask_tensor = Image.fromarray(transformed['gt_fg_mask'])
        gt_fg_mask_tensor = self.clip_mask_transform(gt_fg_mask_tensor)
        gt_fg_mask_tensor = torch.where(gt_fg_mask_tensor > 0.5, 1, 0)
        fg_img_tensor = Image.fromarray(transformed['fg_img'])
        fg_img_tensor = self.clip_transform(fg_img_tensor)
        indicator_tensor = torch.tensor(indicator, dtype=torch.int32)
        # get background image
        inpaint = gt_img_tensor * (mask_tensor < 0.5)
        # del transformed
        # transformed = None
        return {"gt_img":  gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "gt_bbox": gt_bbox_tensor,
                "bg_img": inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "gt_fg_mask": gt_fg_mask_tensor,
                "bbox": bbox_tensor,
                "indicator": indicator_tensor}

    def replace_background_in_foreground(self, fg_img, fg_mask, index):
        obj_idx = int((np.random.randint(1, 1000) + index) % self.length)
        obj_id  = self.sample_list[obj_idx]
        bg_name = list(self.sample_dict[obj_id]['image'].keys())[0]
        # get source image and mask
        image_path = os.path.join(self.dataset_dir, obj_id, 'image', bg_name)
        bg_img = read_pcache_image(image_path).convert('RGB')
        bg_width, bg_height = bg_img.size
        fg_height, fg_width = fg_img.shape[:2]
        if bg_width < fg_width or bg_height < fg_height:
            scale = max(float(fg_width) / bg_width, float(fg_height) / bg_height)            
            bg_width  = int(math.ceil(scale * bg_width))
            bg_height = int(math.ceil(scale * bg_height))
            bg_img = bg_img.resize((bg_width, bg_height), Image.BICUBIC)
        bg_crop = random_crop_image(bg_img, fg_width, fg_height)
        fg_img  = np.where(fg_mask[:,:,np.newaxis] >= 127, fg_img, bg_crop)
        return fg_img, bg_crop

    def __getitem__(self, index):
        # try:
        # randomly sample an object
        obj_id   = self.sample_list[index]
        obj_imgs = list(self.sample_dict[obj_id]['image'].keys())
        # randomly sample two images, we take one as source image and the other as foreground
        fg_name, src_name = random.sample(obj_imgs, 2)
        # process foreground image
        fg_path = os.path.join(self.dataset_dir, obj_id, 'image', fg_name)
        orifg_img  = read_pcache_image(fg_path, 'pil')
        orifg_img  = np.asarray(orifg_img)

        fgmask_path = os.path.join(self.dataset_dir, obj_id, 'mask', fg_name.replace('.jpg', '.png'))
        fgmask = read_pcache_image(fgmask_path, 'cv2')
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        
        fgbbox_path = os.path.join(self.dataset_dir, obj_id, 'bbox', fg_name.replace('.jpg', '.txt'))
        fgbbox = self.load_bbox_file(fgbbox_path)
        fg_img, fg_mask, fgbbox  = crop_foreground_by_bbox(orifg_img, fgmask, fgbbox)

        # process background image
        source_path = os.path.join(self.dataset_dir, obj_id, 'image', src_name)
        source_img  = read_pcache_image(source_path, 'pil') 
        source_np   = np.asarray(source_img)

        mask_path   = os.path.join(self.dataset_dir, obj_id, 'mask', src_name.replace('.jpg', '.png'))
        mask        = read_pcache_image(mask_path, 'cv2')
        mask        = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        bbox_path   = os.path.join(self.dataset_dir, obj_id, 'bbox', src_name.replace('.jpg', '.txt'))
        bbox = self.load_bbox_file(bbox_path)
        src_fgimg, src_fgmask, bbox = crop_foreground_by_bbox(source_np, mask, bbox)

        if self.replace_background_prob > 0 and np.random.rand() < self.replace_background_prob:
            # replace the background of foreground image
            fg_img, new_bg = self.replace_background_in_foreground(fg_img, fg_mask, index)
            src_fgimg, src_newbg = self.replace_background_in_foreground(src_fgimg, src_fgmask, index)
        else:
            new_bg = None
            src_newbg = None
        
        # perform data augmentation
        if self.sample_mode == 'random':
            if max(self.sample_prob) == 1:
                augment_type = self.sample_prob.index(1)
            else:
                augment_type = np.random.choice(self.augment_list, 1, p=self.sample_prob)[0]
                augment_type = int(augment_type)
            indicator = self.augment_types[augment_type]
            sample = self.sample_one_augmentations(source_np, bbox, mask, 
                                                   fg_img, fg_mask, indicator, new_bg,
                                                   src_fgimg, src_fgmask, src_newbg)
        else:
            sample = self.sample_all_augmentations(source_np, bbox, mask, 
                                                   fg_img, fg_mask, new_bg, 
                                                   src_fgimg, src_fgmask, src_newbg)
        sample['image_path'] = source_path
        return sample
        # except:
        #     idx = np.random.randint(0, len(self)-1)
        #     return self[idx]
        
    def __len__(self):
        return self.length

class MVDataAugmentation:
    # data augmentation for multi-view data
    def __init__(self, augment_background, augment_bbox, border_mode=0):
        self.blur = A.Blur(p=0.3)
        self.appearance_trans = A.Compose([
            A.ColorJitter(brightness=0.5, 
                          contrast=0.5, 
                          saturation=0.5, 
                          hue=0.05, 
                          always_apply=False, 
                          p=1)],
            additional_targets={'image':'image', 'image1':'image', 'image2':'image', 'image3': 'image'}
            )
        self.geometric_trans = A.Compose([
            A.HorizontalFlip(p=0.15),
            A.Rotate(limit=30,
                     border_mode=border_mode,
                     value=(127,127,127),
                     mask_value=0,
                     p=0.5),
            A.Perspective(scale=(0., 0.1), 
                          pad_mode=border_mode,
                          pad_val =(127,127,127),
                          mask_pad_val=0,
                          fit_output=False, 
                          p=0.5)
        ])
        self.crop_bg_p  = 0.5
        self.pad_bbox_p = 0.5 if augment_bbox else 0
        self.augment_background_p = 0.3 if augment_background else 0
        self.bbox_maxlen = 0.7
    
    def __call__(self, bg_img, bbox, bg_mask, 
                 fg_img, fg_mask, indicator, new_bg,
                 src_fgimg, src_fgmask, src_newbg):
        # randomly crop background image
        if self.crop_bg_p > 0 and np.random.rand() < self.crop_bg_p:
            crop_bg, crop_bbox, crop_mask = self.random_crop_background(bg_img, bbox, bg_mask)
        else:
            crop_bg, crop_bbox, crop_mask = bg_img, bbox, bg_mask
        # randomly pad bounding box of foreground
        if self.pad_bbox_p > 0 and np.random.rand() < self.pad_bbox_p:
            pad_bbox = self.random_pad_bbox(crop_bbox, crop_bg.shape[1], crop_bg.shape[0])
        else:
            pad_bbox = crop_bbox
        pad_mask = bbox2mask(pad_bbox, crop_bg.shape[1], crop_bg.shape[0])
        # perform illumination transformation on background
        if indicator[0] == 0 and self.augment_background_p > 0 and np.random.rand() < self.augment_background_p:
            trans_imgs = self.appearance_trans(image=crop_bg.copy())
            trans_bg = trans_imgs['image']
        else:
            trans_bg = crop_bg.copy()
        # perform illumination and pose transformation on foreground
        if indicator[1] == 0:
            # replace foreground with the one from the same source image as background
            fg_img, fg_mask, new_bg = src_fgimg, src_fgmask, src_newbg
        app_trans_fg, geo_trans_fg, trans_fgmask, app_src_fg = self.augment_foreground(fg_img.copy(), fg_mask.copy(), indicator, 
                                                                                       new_bg, src_fgimg.copy())
        trans_fg = app_trans_fg if indicator[1] == 0 else geo_trans_fg
        # generate composite by copy-and-paste foreground object
        if indicator[0] == 0:
            x1,y1,x2,y2 = crop_bbox
            trans_bg[y1:y2,x1:x2] = np.where(src_fgmask[:,:,np.newaxis] > 127, app_src_fg, trans_bg[y1:y2,x1:x2])
        transformed = self.blur(image=trans_fg)
        trans_fg = transformed['image']
        transformed = None
        return {"bg_img":   trans_bg,
                "bg_mask":  crop_mask,
                "bbox":     crop_bbox,
                "pad_bbox": pad_bbox,
                "pad_mask": pad_mask,
                "fg_img":   trans_fg,
                "fg_mask":  trans_fgmask,
                "gt_fg_mask": fg_mask}
    
    # @func_set_timeout(0.1)
    def perform_geometry_augmentation(self, app_img, trans_mask, new_bg):
        # geometric transformed image
        transformed = self.geometric_trans(image=app_img, mask=trans_mask)
        geo_img    = transformed['image']
        trans_mask = transformed['mask']
        if new_bg is not None:
            geo_img = np.where(trans_mask[:,:,np.newaxis] > 127, geo_img, new_bg)
        return geo_img, trans_mask
    
    def augment_foreground(self, img, mask, indicator, new_bg, src_img):
        # appearance transformed image
        if new_bg is None:
            transformed = self.appearance_trans(image=img, image1=src_img)
        else:
            transformed = self.appearance_trans(image=img, image1=src_img, image2=new_bg)
            new_bg = transformed['image2']
        app_img = transformed['image']
        app_srcimg = transformed['image1']

        if indicator[1] == 1:
            geo_img, trans_mask = self.perform_geometry_augmentation(app_img, mask, new_bg)
        else:
            geo_img = img
            trans_mask = mask
        return app_img, geo_img, trans_mask, app_srcimg

    def random_crop_background(self, image, bbox, mask):
        width, height = image.shape[1], image.shape[0]
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        # inpaint_w, inpaint_h = inpaint.shape[1], inpaint.shape[0]
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
        trans_image = image[top:down, left:right]
        # trans_inpaint = inpaint[top:down, left:right]
        trans_mask  = mask[top:down, left:right]
        return trans_image, trans_bbox, trans_mask
    
    def random_pad_bbox(self, bbox, width, height):
        bbox_pad  = bbox.copy()
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        if bbox_w < self.bbox_maxlen:
            maxpad = width * min(self.bbox_maxlen - bbox_w, bbox_w) * 0.5
            bbox_pad[0] = max(0, int(bbox[0] - np.random.rand() * maxpad))
            bbox_pad[2] = min(width, int(bbox[2] + np.random.rand() * maxpad))
        
        if bbox_h < self.bbox_maxlen:
            maxpad = height * min(self.bbox_maxlen - bbox_h, bbox_h) * 0.5
            bbox_pad[1] = max(0, int(bbox[1] - np.random.rand() * maxpad))
            bbox_pad[3] = min(height, int(bbox[3] + np.random.rand() * maxpad))
        return bbox_pad

def test_mvimgnet_dataset():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_multidata.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    dataset_list = []
    for i in range(len(configs)):
        configs[i].params.split = 'train'
        configs[i].params.augment_config.sample_mode = 'all' 
        configs[i].params.augment_config.augment_types = [(0,0), (1,0), (0,1), (1,1)]
        single_dataset = instantiate_from_config(configs[i])
        dataset_list.append(single_dataset)
        print('{}-th dataset contains {} samples'.format(i, len(single_dataset)))
    dataset = ConcatDataset(dataset_list)
    aug_cfg = configs[-1].params.augment_config
    bs = 1 if aug_cfg.sample_mode == 'all' else 4
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=True,
                            num_workers=0)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    exit()
    vis_dir = os.path.join(proj_dir, 'outputs/test_dataaug/union_dataset')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == 1:
                batch[k] = batch[k][0]

        file = batch['image_path']
        gt_t = batch['gt_img']
        gtmask_t = batch['gt_mask']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        im_name = os.path.basename(file[0])
        print(i, len(dataloader), gt_t.shape, gtmask_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        print(os.path.join(vis_dir, f'batch{i}.jpg'))
        if i > 30:
            break
    
def test_mvimgnet_efficiency():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_multidata.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    dataset_list = []
    for i in range(len(configs)):
        configs[i].params.split = 'train'
        configs[i].params.augment_config.sample_mode = 'all' 
        configs[i].params.augment_config.augment_types = [(0,0), (1,0), (0,1), (1,1)]
        single_dataset = instantiate_from_config(configs[i])
        dataset_list.append(single_dataset)
        print('{}-th dataset contains {} samples'.format(i, len(single_dataset)))
    dataset = ConcatDataset(dataset_list)
    aug_cfg = configs[-1].params.augment_config
    bs = 16
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=True,
                            num_workers=16)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    start = time.time()
    for i,batch in enumerate(dataloader):
        images = batch['image_path']
        print(images)
        end = time.time()
        if i % 10 == 0:
            print('{}, avg time {:.1f}ms'.format(
                i, (end-start) / (i+1) * 1000
            ))
        
if __name__ == '__main__':
    # test_mvimgnet_dataset()
    test_mvimgnet_efficiency()
    
    
    


