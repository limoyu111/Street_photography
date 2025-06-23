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

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def prepare_input(batch, model, shape, device, augment_types=[[0,0]]):
    num_samples = len(augment_types)
    if num_samples > 1:
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = torch.cat([batch[k]] * num_samples, dim=0)
    test_model_kwargs={}
    bg_img = batch['bg_img'].to(device)
    bg_latent = model.encode_first_stage(bg_img)
    bg_latent = model.get_first_stage_encoding(bg_latent).detach()
    test_model_kwargs['bg_latent'] = bg_latent
    rs_mask = F.interpolate(batch['bg_mask'].to(device), shape[-2:])
    rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
    test_model_kwargs['bg_mask']  = rs_mask
    test_model_kwargs['bbox']  = batch['bbox'].to(device)
    fg_tensor = batch['fg_img'].to(device) # .to(torch.float16)
    fg_mask = batch['fg_mask'].to(device)
    c = model.get_learned_conditioning((fg_tensor, fg_mask))
    indicator = torch.tensor(augment_types).int().to(device)
    # if len(augment_types) == 1:
    #     indicator = torch.tensor(augment_types).repeat(4,1).int().to(device)
    # else:
    #     indicator = torch.tensor([[0,0], [0,1], [1,0], [1,1]]).int().to(device)
    c.append(indicator)
    return test_model_kwargs, c

def get_denoise_row_from_list(samples, model, device, desc='', force_no_decoder_quantization=False):
    denoise_row = []
    for zd in tqdm(samples, desc=desc):
        dz = model.decode_first_stage(zd.to(device),force_not_quantize=force_no_decoder_quantization)
        dz = Resize((256, 256))(dz)
        denoise_row.append(dz)
    n_imgs_per_row = len(denoise_row)
    denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
    denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
    denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    return denoise_grid

def get_mask_row_from_list(samples, desc=''):
    denoise_row = []
    for dm in tqdm(samples, desc=desc):
        dm = (dm * 2.0 - 1.0)
        dm = Resize((256, 256))(dm)
        denoise_row.append(dm)
    n_imgs_per_row = len(denoise_row)
    denoise_row  = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
    denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
    denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    return denoise_grid

def clip2sd(x):
    # clip input tensor to  stable diffusion tensor
     MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1,-1,1,1).to(x.device)
     STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1,-1,1,1).to(x.device)
     denorm = x * STD + MEAN
     sd_x = denorm * 2 - 1
     return sd_x

def tensor2images(images, max_images=4, clamp=True, target_size=(256, 256)):
    for k in images:
        N = min(images[k].shape[0], max_images)
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu().float()
            if clamp:
                images[k] = torch.clamp(images[k], -1., 1.)
    return images

def log_local(save_dir,
              images,
              prefix):
    root = save_dir
    for k in images:
        # grid = torchvision.utils.make_grid(images[k], nrow=images[k].shape[0])
        grid = make_grid(images[k], 1)
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = f'{prefix}-{k}.jpg'
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)
        print(path)

exp_root = 'experiments/finetune_paint/indicator4/'
exp_name = '2023-06-29T17-20-50'
ckpt_dir = exp_root + exp_name + '_Refine/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="intermediate_results/x_T"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        type=bool,
        # action='store_true',
        default=True,
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=ckpt_dir+"configs/"+exp_name+"-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        default=ckpt_dir+"checkpoints/last.ckpt", # last.ckpt
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=ckpt_dir+"/code",
        help="path to code",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)
    
    config = OmegaConf.load(f"{opt.config}")
    augment_config = config.data.params.train.params.augment_config
    # augment_types  = augment_config.augment_types
    augment_types = [[1,1]] * 4
    opt.n_samples  = len(augment_types)
    code_dir = opt.code_dir
    sys.path.insert(0, code_dir)

    model  = load_model_from_config(config, f"{opt.ckpt}")
    print('complete model building')
    model  = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
        opt.ddim_steps = 50
    else:
        sampler = DDIMSampler(model)
        opt.ddim_steps = 200
    
    if os.path.exists(opt.outdir):
        shutil.rmtree(opt.outdir)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    n_rows = opt.n_rows if opt.n_rows > 0 else 1
    data_root  = '../../dataset/cocoee'
    
    dataset = COCOEEDataset(dataset_dir=data_root, 
                               image_size=512,
                               split='test')
    dataloader = DataLoader(dataset, 
                            1, 
                            False, 
                            num_workers=4)
    # src_dir    = os.path.join(data_root, "GT_3500")
    # ref_dir    = os.path.join(data_root, 'Ref_3500')
    # mask_dir   = os.path.join(data_root, 'Mask_bbox_3500')  
    
    # result_path = os.path.join(outpath, "results")
    # grid_path = os.path.join(outpath, "grid")
    # os.makedirs(result_path, exist_ok=True)
    # os.makedirs(grid_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        print('use fixed start code ...')
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for index, batch in enumerate(dataloader):
                    image_name = os.path.splitext(os.path.basename(batch['image_path'][0]))[0]
                    if image_name not in  ['000001051688_GT', '000000000283_GT', '000000354418_GT', 
                                           '000000431812_GT', '000000353634_GT', '000000213111_GT',
                                           '000000300317_GT', '000000360363_GT', '000000423037_GT',
                                           '000000002447_GT', '000000002447_GT', '000000023028_GT',
                                           '000000002187_GT', '000000004421_GT', '000000003658_GT',
                                           '000001204623_GT', '000000431114_GT', '000000169920_GT']:
                        continue
                    start = time.time()
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    test_model_kwargs, c = prepare_input(batch, model, shape, device, augment_types)
                    # ts = torch.full((test_model_kwargs['bg_latent'].shape[0],), 999, 
                    #                  device=test_model_kwargs['bg_latent'].device, 
                    #                  dtype=torch.long)
                    # start_code = model.q_sample(test_model_kwargs['bg_latent'], t=ts) 
                    uc = None
                    if opt.scale != 1.0 and model.use_guidance:
                        uc = model.learnable_vector
                    samples_ddim, z_denoise_row = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    eta=opt.ddim_eta,
                                                    mask=None,
                                                    x_T=start_code,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    test_model_kwargs=test_model_kwargs)
                    # pre_mask = F.interpolate(samples_ddim[:,4:], (opt.H, opt.W))
                    x_samples_ddim = model.decode_first_stage(samples_ddim[:,:4]).cpu().float()

                    denoise_grid = get_denoise_row_from_list(z_denoise_row['x_inter'], model, device)
                    predx0_grid  = get_denoise_row_from_list(z_denoise_row['pred_x0'], model, device)
                    # mask_grid    = get_mask_row_from_list(z_denoise_row['mask_inter'])
                    log = dict()
                    img_size = (256, 256)
                    masked_bg = batch['bg_img'] * (batch['bg_mask'] < 0.5) 
                    cat_img  = [
                        Resize(img_size)(batch['bg_img']),
                        Resize(img_size)(masked_bg),
                        Resize(img_size)(clip2sd(batch['fg_img'])),
                        Resize(img_size)(x_samples_ddim),
                        ]
                    log['samples'] = torch.cat(cat_img, dim=-1)
                    images = tensor2images(log)
                    log_local(outpath, images, image_name)
                    time_cost = time.time() - start
                    print('inference {} images, time cost {:.1f} s'.format(batch['bg_img'].shape[0], time_cost))
                    # if index > 100:
                    #     break
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
