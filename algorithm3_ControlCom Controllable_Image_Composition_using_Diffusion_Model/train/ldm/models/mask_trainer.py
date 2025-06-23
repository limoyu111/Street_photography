from genericpath import samefile
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize, Normalize
import math
import time
import random
from torch.autograd import Variable
from ldm.modules.mask_blur import GaussianBlurMask
from ldm.modules.losses import DiceBCELoss, BCELoss, DiceLoss
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from ldm.modules.diffusion_metrics import CustomCLIPScore

class MaskGenerator(pl.LightningModule):
    def __init__(self,
                 mask_generator_config,
                 diffusion_config,
                 mask_loss_type='BCE'):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diffusion_config)
        
