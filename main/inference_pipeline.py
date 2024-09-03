"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
Author : Yu Yamaoka(Osaka Univ)
mail:yu-yamaoka@ist.osaka-u.ac.jp)
"""

import os, sys
import json
from pathlib import Path
from ml_collections import config_dict

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

# Set up GPU environment manually
class Args:
    dist_url = "env://"
    local_rank = 0

args = Args()
utils.init_distributed_mode(args)

cudnn.benchmark = True

# Config for model parameters
model_param = config_dict.ConfigDict()
model_param.arch = "vit_base"
model_param.patch_size = 8
model_param.n_last_blocks = 1
model_param.avgpool_patchtokens = True #PreTrainModel is True，own model is False． Should set to True for Vit-Base according to the authors of Dino?
model_param.checkpoint_key = "student"
model_param.fc_out_dim = 1

# Initialize Linear Regressor Architecture
class LinearRegression(nn.Module):
    def __init__(self, dim, out_dim=1):
        super().__init__() # python3 dose not need args
        self.fc1 = nn.Linear(dim, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.fc3 = nn.Linear(64, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y

class MLP(nn.Module):
    def __init__(self, dim, out_dim=1):
        super().__init__() # python3 dose not need args
        self.fc1 = nn.Linear(dim, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.fc3 = nn.Linear(64, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y

# Initialize OSLLP ViT Model
def load_models(vit_model_path, fc_model_path):
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if model_param.arch in vits.__dict__.keys():
        model = vits.__dict__[model_param.arch](patch_size=model_param.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (model_param.n_last_blocks + int(model_param.avgpool_patchtokens))    
    # if the network is a XCiT
    elif "xcit" in model_param.arch:
        model = torch.hub.load('facebookresearch/xcit:main', model_param.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif model_param.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[model_param.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {model_param.arch}")
        sys.exit(1)
    print(embed_dim,"***************::")
    model.cuda()
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    vit_model = nn.parallel.DistributedDataParallel(model, device_ids=[0])
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, vit_model_path, model_param.checkpoint_key, model_param.arch, model_param.patch_size)
    print(f"Model {model_param.arch} built.")

    linear_regressor = LinearRegression(embed_dim, out_dim=model_param.fc_out_dim)
    linear_regressor = linear_regressor.cuda()
    linear_regressor = nn.parallel.DistributedDataParallel(linear_regressor, device_ids=[0])
    # Load linear_regressor weights
    fc_state_dict = torch.load(fc_model_path, map_location="cpu")["state_dict"]
    msg = linear_regressor.load_state_dict(fc_state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(fc_model_path, msg))
   
    return vit_model, linear_regressor

# Initialize OSLLP ViT Model
def load_models_classifier(vit_model_path, fc_model_path, class_num):
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if model_param.arch in vits.__dict__.keys():
        model = vits.__dict__[model_param.arch](patch_size=model_param.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (model_param.n_last_blocks + int(model_param.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in model_param.arch:
        model = torch.hub.load('facebookresearch/xcit:main', model_param.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif model_param.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[model_param.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {model_param.arch}")
        sys.exit(1)
    model.cuda()
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    vit_model = nn.parallel.DistributedDataParallel(model, device_ids=[0])
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, vit_model_path, model_param.checkpoint_key, model_param.arch, model_param.patch_size)
    print(f"Model {model_param.arch} built.")

    #load head of model
    llp_classifier = MLP(embed_dim, out_dim=class_num)
    llp_classifier = llp_classifier.cuda()
    llp_classifier = nn.parallel.DistributedDataParallel(llp_classifier, device_ids=[0])
    # Load llp_classifier weights
    fc_state_dict = torch.load(fc_model_path, map_location="cpu")["state_dict"]
    msg = llp_classifier.load_state_dict(fc_state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(fc_model_path, msg))
   
    return vit_model, llp_classifier

# OSLLP ViT inference
@torch.no_grad()
def feature_extractor(model, img, model_param=model_param):
    # Define transform for image
    transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Resize(64),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
  
    # Do image transform and move to GPU
    img = transform(img)
    assert torch.is_tensor(img), "The image must be a PyTorch tensor."
    img = img.cuda(non_blocking=True).unsqueeze(0)
  
    with torch.no_grad():
        if "vit" in model_param.arch:
            intermediate_output = model.module.get_intermediate_layers(img, model_param.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if model_param.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = model(img)

    return output
      
@torch.no_grad()
# Linear Regressor inference
def linear_regression(regressor, feat):
    """
    return : class(float)
    """
    with torch.no_grad():
        output = regressor(feat)
    
    return output

# Linear Regressor inference
def classifier(classifier, feat):
    """
    return : class(int)
    """
    with torch.no_grad():
        output = classifier(feat)
    
    return torch.argmax(output)

