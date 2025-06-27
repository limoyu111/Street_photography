import torch
from omegaconf import OmegaConf
import sys,os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)
from ldm.util import instantiate_from_config

def add_input_channels(num_ch=1):
    weight     = ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'] 
    zero_data  = torch.zeros(weight.shape[0], num_ch, 3, 3)
    new_weight = torch.cat((weight, zero_data), dim=1)
    ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'] = new_weight

def add_output_channels(num_ch=1):
    weight  = ckpt_file['state_dict']['model.diffusion_model.out.2.weight'] 
    zeros_weight = torch.zeros(num_ch, weight.shape[1], 3, 3)
    new_weight = torch.cat([weight, zeros_weight], dim=0)
    ckpt_file['state_dict']['model.diffusion_model.out.2.weight'] = new_weight
    
    bias = ckpt_file['state_dict']['model.diffusion_model.out.2.bias']
    zeros_bias = torch.zeros(num_ch)
    new_bias   = torch.cat([bias, zeros_bias], dim=0)
    ckpt_file['state_dict']['model.diffusion_model.out.2.bias'] = new_bias
    
def change_specific_number(name, index, offset=1):
    ch = list(name.split('.'))
    for i in range(len(ch)):
        if ch[i].isdigit():
            index -= 1
            if index == 0: 
                ch[i] = str(int(ch[i]) + offset)
                break
    return '.'.join(ch)

def get_specific_number(name, index=0):
    ch = list(name.split('.'))
    for i in range(len(ch)):
        if ch[i].isdigit():
            index -= 1
            if index == 0:
                return int(ch[i])
            
    
def add_middle_local_attention_module(model):
    new_weights = dict()
    for name,weights in model.named_parameters():
        if 'model.diffusion_model.middle_block' in name:
            if get_specific_number(name, 1) == 0:
                new_weights[name] = ckpt_file['state_dict'].pop(name)
                
            elif get_specific_number(name, 1) == 1:
                new_weights[name] = weights
            else:
                src_name = change_specific_number(name, 1, -1)
                new_weights[name] = ckpt_file['state_dict'].pop(src_name)
    for k,v in new_weights.items():
        ckpt_file['state_dict'][k] = v
        
def add_output_local_attention_module(model):
    localatt_blocks = []
    postconv_layers = []
    for name,weights in model.named_parameters():
        if 'model.diffusion_model.output_blocks' in name and 'proj_in.weight' in name and weights.shape[1] % 10 == 2:
            localatt_blocks.append(name.split('.proj_in.weight')[0])
            conv = change_specific_number(name, 2, 1).replace('proj_in.weight', 'conv.weight')
            
    for name,weights in model.named_parameters():
        for block in localatt_blocks:
            if block in name:
                if name in ckpt_file['state_dict']:
                    source_weight = ckpt_file['state_dict'][name]
                    new_name = change_specific_number(name, 2, 1)
                    ckpt_file['state_dict'][new_name] = source_weight
                ckpt_file['state_dict'][name] = weights
    conv_layers = ['model.diffusion_model.output_blocks.5.2.conv.weight', 
                   'model.diffusion_model.output_blocks.5.2.conv.bias', 
                   'model.diffusion_model.output_blocks.8.2.conv.weight', 
                   'model.diffusion_model.output_blocks.8.2.conv.bias']
    for name in conv_layers:
        new_name = change_specific_number(name, 2, 1)
        ckpt_file['state_dict'][new_name] = ckpt_file['state_dict'].pop(name)
                
                
def add_input_local_attention_module(model):
    localatt_blocks = []
    for name,weights in model.named_parameters():
        if 'model.diffusion_model.input_blocks' in name and 'proj_in.weight' in name and weights.shape[1] % 10 == 2:
            localatt_blocks.append(name.split('.proj_in.weight')[0])
    
    for name,weights in model.named_parameters():
        for block in localatt_blocks:
            if block in name:
                if name in ckpt_file['state_dict']:
                    source_weight = ckpt_file['state_dict'][name]
                    new_name = change_specific_number(name, 2, 1)
                    ckpt_file['state_dict'][new_name] = source_weight
                ckpt_file['state_dict'][name] = weights
    
def build_model(config):
    return instantiate_from_config(config.model)
    

def load_model_from_config(model, verbose=True):
    sd = ckpt_file["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model


pretrained_model_path='pretrained_models/sd-v1-4.ckpt'
print(f"Loading model from {pretrained_model_path}")
ckpt_file=torch.load(pretrained_model_path,map_location='cpu')
new_model_path = 'pretrained_models/sd-v1-4-modified.ckpt'
config = OmegaConf.load('configs/v1.yaml')
model = build_model(config)
add_input_channels()
add_output_channels()
add_input_local_attention_module(model)
add_middle_local_attention_module(model)
add_output_local_attention_module(model)
torch.save(ckpt_file, new_model_path)
model = load_model_from_config(model)