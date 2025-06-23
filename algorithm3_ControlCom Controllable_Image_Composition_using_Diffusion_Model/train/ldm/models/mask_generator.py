# borrow heavily from https://github.com/Lipurple/Grounded-Diffusion/blob/main/ldm/models/seg_module.py
from functools import partial
import math
from typing import Iterable
from torch import nn, einsum
import numpy as np
import torch as th
import functools
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]

# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                which_conv=nn.Conv2d, which_linear=None, activation=None, 
                upsample=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
       
        self.register_buffer('stored_mean1', torch.zeros(in_channels))
        self.register_buffer('stored_var1',  torch.ones(in_channels)) 
        self.register_buffer('stored_mean2', torch.zeros(out_channels))
        self.register_buffer('stored_var2',  torch.ones(out_channels)) 
        
        self.upsample = upsample

    def forward(self, x, y=None):
        x = F.batch_norm(x, self.stored_mean1, self.stored_var1, None, None,
                          self.training, 0.1, 1e-4)
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = F.batch_norm(h, self.stored_mean2, self.stored_var2, None, None,
                          self.training, 0.1, 1e-4)
        
        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x

class MaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        low_feature_channel = 16
        mid_feature_channel = 32
        high_feature_channel = 64
        highest_feature_channel=128
        
        self.low_feature_conv = nn.Sequential(
            nn.Conv2d(1280*6*2, low_feature_channel, kernel_size=1, bias=False),
        )
        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d((1280*5+640)*2, mid_feature_channel, kernel_size=1, bias=False),
        )
        self.mid_feature_mix_conv = SegBlock(
                                in_channels=low_feature_channel+mid_feature_channel,
                                out_channels=low_feature_channel+mid_feature_channel,
                                con_channels=128,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        self.high_feature_conv = nn.Sequential(
            nn.Conv2d((1280+640*4+320)*2, high_feature_channel, kernel_size=1, bias=False),
        )
        self.high_feature_mix_conv = SegBlock(
                                in_channels=low_feature_channel+mid_feature_channel+high_feature_channel,
                                out_channels=low_feature_channel+mid_feature_channel+high_feature_channel,
                                con_channels=128,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        self.highest_feature_conv = nn.Sequential(
            nn.Conv2d((640+320*6)*2, highest_feature_channel, kernel_size=1, bias=False),
        )
        self.highest_feature_mix_conv = SegBlock(
                                in_channels=low_feature_channel+mid_feature_channel+high_feature_channel+highest_feature_channel,
                                out_channels=low_feature_channel+mid_feature_channel+high_feature_channel+highest_feature_channel,
                                con_channels=128,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        
        feature_dim=low_feature_channel+mid_feature_channel+high_feature_channel+highest_feature_channel
        self.conv_final = nn.Conv2d(feature_dim, 1, 1, 1, 0)
        
    def forward(self,diffusion_feature):
        image_feature=self._prepare_features(diffusion_feature)
        final_image_feature=F.interpolate(image_feature, size=512, mode='bilinear', align_corners=False)
        pre_mask = torch.sigmoid(self.conv_final(final_image_feature))
        return pre_mask

    def _prepare_features(self, features, upsample='bilinear'):
        self.low_feature_size = 16
        self.mid_feature_size = 32
        self.high_feature_size = 64
        
        low_features = [
            F.interpolate(i, size=self.low_feature_size, mode=upsample, align_corners=False) for i in features['low']
        ]
        low_features = torch.cat(low_features, dim=1)
        
        mid_features = [
             F.interpolate(i, size=self.mid_feature_size, mode=upsample, align_corners=False) for i in features['mid']
        ]
        mid_features = torch.cat(mid_features, dim=1)
        
        high_features = [
             F.interpolate(i, size=self.high_feature_size, mode=upsample, align_corners=False) for i in features['high']
        ]
        high_features = torch.cat(high_features, dim=1)
        highest_features=torch.cat(features["highest"],dim=1)
        features_dict = {
            'low': low_features,
            'mid': mid_features,
            'high': high_features,
            'highest':highest_features,
        }

        low_feat = self.low_feature_conv(features_dict['low'])
        low_feat = F.interpolate(low_feat, size=self.mid_feature_size, mode='bilinear', align_corners=False)
        
        mid_feat = self.mid_feature_conv(features_dict['mid'])
        mid_feat = torch.cat([low_feat, mid_feat], dim=1)
        mid_feat = self.mid_feature_mix_conv(mid_feat, y=None)
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode='bilinear', align_corners=False)
        
        high_feat = self.high_feature_conv(features_dict['high'])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat, y=None)
        
        highest_feat=self.highest_feature_conv(features_dict['highest'])
        highest_feat=torch.cat([high_feat,highest_feat],dim=1)
        highest_feat=self.highest_feature_mix_conv(highest_feat,y=None)
        
        return highest_feat