from kornia.filters import gaussian_blur2d
import copy
import torch

class GaussianBlurMask:
    def __init__(self, max_step=1000):
        self.max_step = max_step
        
    def get_mask_coef(self, step, x0=0.8, y0=0.2):
        '''
        compute a coefficient to scale the Guassian kernel size
        '''
        x = torch.clip(step.float() / self.max_step, min=0., max=1.)
        # coef1 = (x / x0 * y0).float()
        # coef2 = (y0 + (x - x0) / (1 - x0) * (1 - y0)).float()
        # coef  = torch.where(x < x0, coef1, coef2)
        coef = x
        return coef
    
    def check_kernel_size(self, ks, max_ks):
        '''
        make sure the kernel size is an odd number and does not exceed the upper bound.
        '''
        ks = ks.int()
        ks = torch.where(ks % 2 == 0, ks + 1, ks)
        ks = torch.where(ks > max_ks, max_ks, ks)
        return ks
        
    def __call__(self, mask, bbox, step):
        # compute the kernel size and sigma of gaussian blur
        mask_h, mask_w = mask.shape[-2:]
        bbox_int = copy.deepcopy(bbox)
        bbox_int[:, 0::2] *= mask_w
        bbox_int[:, 1::2] *= mask_h
        bbox_int = torch.round(bbox_int).int()
        bbox_w = bbox_int[:,2] - bbox_int[:,0]
        bbox_h = bbox_int[:,3] - bbox_int[:,1]
        max_kw = bbox_w * 2 - 1
        max_kh = bbox_h * 2 - 1
        coef = self.get_mask_coef(step)
        kernel_w  = self.check_kernel_size(coef * max_kw, max_kw)
        kernel_h  = self.check_kernel_size(coef * max_kh, max_kh)
        kernel_size = torch.stack([kernel_h, kernel_w], dim=1)
        sigma_w   = kernel_w.float() / 3
        sigma_h   = kernel_h.float() / 3
        # gaussian blur each mask in a batch
        sigma = torch.stack([sigma_h, sigma_w], dim=1)
        mask_blur = []
        for i in range(mask.shape[0]):
            ks = (kernel_size[i,0].item(), kernel_size[i,1].item())
            ret = self.blur_mask(mask[i:i+1], bbox_int[i], ks, sigma[i])
            mask_blur.append(ret)
        mask_blur = torch.cat(mask_blur, dim=0)
        return mask_blur
                
    def blur_mask(self, mask, bbox, kernel_size, sigma):
        '''
        gaussian blur a region of input mask according to the bbox
        '''
        x1, y1, x2, y2 = bbox
        out_mask = copy.deepcopy(mask)
        local_mask = out_mask[:, :, y1:y2, x1:x2]
        local_mask = gaussian_blur2d(local_mask, kernel_size, sigma)
        local_mask = torch.where(local_mask > 1e-5, 1., 0.).float()
        out_mask[:, :, y1:y2, x1:x2] = local_mask
        return out_mask