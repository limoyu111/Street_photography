import torch
from omegaconf import OmegaConf
import sys,os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)
from ldm.util import instantiate_from_config

def add_condition_layers():
    origin_weight = 'proj_out.weight'
    target_weight = 'cond_stage_model.proj_out.weight'
    ckpt_file['state_dict'][target_weight] = ckpt_file['state_dict'].pop(origin_weight)
    origin_bias = 'proj_out.bias'
    target_bias = 'cond_stage_model.proj_out.bias'
    ckpt_file['state_dict'][target_bias] = ckpt_file['state_dict'].pop(origin_bias)
    # ckpt_file['state_dict'].pop('model.diffusion_model.local_positional_embedding')

def add_input_channels(num_ch):
    weight     = ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'] 
    new_add = num_ch - weight.shape[1]
    if new_add > 0:
        zero_data  = torch.zeros(weight.shape[0], new_add, 3, 3)
        new_weight = torch.cat((weight, zero_data), dim=1)
        ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'] = new_weight

def add_output_channels(num_ch):
    weight  = ckpt_file['state_dict']['model.diffusion_model.out.2.weight']
    new_add = num_ch - weight.shape[0] 
    if new_add > 0:
        zeros_weight = torch.zeros(new_add, weight.shape[1], 3, 3)
        new_weight = torch.cat([weight, zeros_weight], dim=0)
        ckpt_file['state_dict']['model.diffusion_model.out.2.weight'] = new_weight
        
        bias = ckpt_file['state_dict']['model.diffusion_model.out.2.bias']
        zeros_bias = torch.zeros(new_add)
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
    
def add_middle_local_module(model):
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
    local_blocks  = []
    global_blocks = [] 
    for name,weights in model.named_parameters():
        if 'model.diffusion_model.output_blocks' in name and 'context_norm.weight' in name:
            local_prefix = name.split('.context_norm.weight')[0]
            gloab_prefix = change_specific_number(local_prefix, 2, -1) 
            local_blocks.append(local_prefix)
            global_blocks.append(gloab_prefix)
    
    for name,weights in model.named_parameters():
        for block in global_blocks:
            if block in name:
                if name in ckpt_file['state_dict']:
                    source_weight = ckpt_file['state_dict'].pop(name)
                    new_name = change_specific_number(name, 2, 1)
                    ckpt_file['state_dict'][new_name] = source_weight
                ckpt_file['state_dict'][name] = weights

def add_output_local_conv_module(model):
    local_blocks  = []
    global_blocks = [] 
    for name,weights in model.named_parameters():
        if 'model.diffusion_model.output_blocks' in name and 'context_norm.weight' in name:
            local_prefix = name.split('.context_norm.weight')[0]
            gloab_prefix = change_specific_number(local_prefix, 2, 1) 
            local_blocks.append(local_prefix)
            global_blocks.append(gloab_prefix)
            
    for name,weights in model.named_parameters():
        skip = False
        for block in global_blocks:
            if block in name:
                old_name = change_specific_number(name, 2, -1)
                if old_name in ckpt_file['state_dict']:
                    source_weight = ckpt_file['state_dict'].pop(old_name)
                    ckpt_file['state_dict'][name] = source_weight
                    skip = True
                    break
        if not skip:
            for block in local_blocks:
                if block in name:
                    ckpt_file['state_dict'][name] = weights
                    break

    conv_layers = ['model.diffusion_model.output_blocks.5.2.conv.weight', 
                   'model.diffusion_model.output_blocks.5.2.conv.bias', 
                   'model.diffusion_model.output_blocks.8.2.conv.weight', 
                   'model.diffusion_model.output_blocks.8.2.conv.bias']
    for name in conv_layers:
        new_name = change_specific_number(name, 2, 1)
        ckpt_file['state_dict'][new_name] = ckpt_file['state_dict'].pop(name)

def add_input_local_conv_module(model):
    local_blocks  = []
    global_blocks = [] 
    for name,weights in model.named_parameters():
        if 'model.diffusion_model.input_blocks' in name and 'context_norm.weight' in name:
            local_prefix  = name.split('.context_norm.weight')[0]
            global_prefix = change_specific_number(local_prefix, 2, 1)
            global_blocks.append(global_prefix)
            local_blocks.append(local_prefix)

    for name,weights in model.named_parameters():
        skip = False
        for block in global_blocks:
            if block in name:
                old_name = change_specific_number(name, 2, -1)
                if old_name in ckpt_file['state_dict']:
                    source_weight = ckpt_file['state_dict'].pop(old_name)
                    ckpt_file['state_dict'][name] = source_weight
                    skip = True
                    break
            if not skip:
                for block in local_blocks:
                    if block in name:
                        ckpt_file['state_dict'][name] = weights
                        break

def remove_positional_embedding(model):
    keep_learnable_vec = False
    for name,weights in model.named_parameters():
        # if 'positional_embedding' in name and name in ckpt_file['state_dict']:
        #     ckpt_file['state_dict'].pop(name)
        if 'learnable_vector' in name:
            keep_learnable_vec = True
    if not keep_learnable_vec:
        ckpt_file['state_dict'].pop('learnable_vector')
    
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


pretrained_model_path='checkpoints/paint_by_example.ckpt'
print(f"Loading model from {pretrained_model_path}")
ckpt_file=torch.load(pretrained_model_path,map_location='cpu')
config = OmegaConf.load('configs/finetune_paint.yaml')
model  = build_model(config)
add_condition_layers()
in_channels  = config.model.params.unet_config.params.in_channels
add_input_channels(in_channels)
out_channels = config.model.params.unet_config.params.out_channels
add_output_channels(out_channels)
# local_type = config.model.params.unet_config.params.local_encoder_config.conditioning_key
new_model_path = 'pretrained_models/paint-{}channels.ckpt'.format(in_channels)
# add_input_local_conv_module(model)
# add_middle_local_module(model)
# add_output_local_conv_module(model)
# remove_positional_embedding(model)
print('save modified model to ', new_model_path)
torch.save(ckpt_file, new_model_path)
model = load_model_from_config(model)