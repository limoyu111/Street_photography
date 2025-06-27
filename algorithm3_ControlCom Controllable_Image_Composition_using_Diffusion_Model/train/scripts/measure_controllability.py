import argparse, os, sys, glob
from cgi import test
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
import sys, os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)
# os.chdir(proj_dir)
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.open_images_control import COCOEEDataset, OpenImageDataset, reverse_image_tensor, reverse_mask_tensor, reverse_clip_tensor
import time
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch.nn import functional as F
import shutil
