import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from PIL import Image
import glob
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformations_torch import *
from model import HorizonNet
    

def warp(src_transformer, target_transformer, target_bon, H, W, ceiling_z): 
    N = target_bon.shape[0]    # (N, 2, 1024)
    device = target_bon.device
    
    theta = ((torch.arange(W).expand(N, W).to(device) / W)) * 2 * np.pi # (N, 1024)
    target_3d_coord = []
    
    z_bons = [ceiling_z, torch.zeros_like(ceiling_z).to(device) - 1.] # (N, 1)
    for i, z_bon in enumerate(z_bons):
        y_start = (H // 2) * i
        y_end = y_start + (H // 2)
        
        y_grid, x_grid = torch.meshgrid(torch.arange(y_start, y_end), torch.arange(W))    
        y_grid, x_grid = y_grid.expand(N, H//2, W), x_grid.expand(N, H//2, W)    # (N, 256, 1024)
        x_grid = x_grid.to(device)
        y_grid = y_grid.to(device)
        
        boundary_phi = (0.5 - (target_bon[:, i] / H)) * np.pi # (N, 1024)
        boundary_dist = torch.abs(z_bon / torch.tan(boundary_phi)) # (N, 1024)

        boundary_x = boundary_dist * torch.sin(theta) # (N, 1024)
        boundary_y = boundary_dist * torch.cos(theta + np.pi) # (N, 1024)
        boundary_R = torch.abs(z_bon / torch.sin(boundary_phi))  # (N, 1024)

        assert not torch.any(torch.isnan(boundary_dist))

        all_phi = (0.5 - (y_grid / H)) * np.pi # (N, 256, 1024)
        all_dist = torch.min(torch.abs(z_bon[:, :, None] / torch.tan(all_phi)), boundary_dist[:, None, :]) # (N, 256, 1024)
        all_x = all_dist * torch.sin(theta[:, None, :]) # (N, 256, 1024)
        all_y = all_dist * torch.cos(theta[:, None, :] + np.pi) # (N, 256, 1024)  
        all_z = torch.max(torch.min(boundary_R[:, None, :] * torch.sin(all_phi), ceiling_z[:, :, None]), torch.Tensor([-1.]).to(device))

        target_3d_coord.append(torch.stack([all_x, all_y, all_z], dim=-1).view(N, -1, 3))    # (N, 256*1024, 3)
    

    target_3d_coord = torch.cat(target_3d_coord, dim=1) # (N, 512*1024, 3)
    global_2d_coord = target_transformer.to_global(target_3d_coord[:, :, :2]) # (N, 512*1024, 2)

    src_2d_coord = src_transformer.apply_inverse(global_2d_coord) # (N, 512*1024, 2)
    src_3d_coord = torch.cat([src_2d_coord, target_3d_coord[:, :, 2:]], dim=-1).view(-1, 3) # (N*512*1024, 3)

    src_pix_coord = TransformationSpherical.cartesian_to_pixel(src_3d_coord, W) # (N*512*1024, 2)
    
    src_x_grid = ((src_pix_coord[:, 0] / W - 0.5) * 2).view(N, H, W) # (N, 512, 1024)
    src_y_grid = ((src_pix_coord[:, 1] / H - 0.5) * 2).view(N, H, W) # (N, 512, 1024)
    
    grid = torch.stack([src_x_grid, src_y_grid], axis=-1)    # (N, 512, 1024, 2)
    
    return grid  

def generate_mask(shape, margin):
    N, C, H, W = shape
    return torch.cat([torch.ones(N, C, H - margin, W), torch.zeros(N, C, margin, W)], axis=2)

def inference_ceiling(y_bon, H, W):
    N, C, _ = y_bon.shape
    local_2d = compute_local(y_bon, H, W, torch.ones(N, 1).to(y_bon.device)) # (N, 2*1024, 2)
    ceil_2d = local_2d[:, :W, :]
    floor_2d = local_2d[:, W:, :]
    
    ceil_dist = torch.norm(ceil_2d, dim=-1)
    floor_dist = torch.norm(floor_2d, dim=-1)
    
    scale = (floor_dist / ceil_dist).mean(dim=-1) 

    return scale

def compute_local(src_bon, H, W, ceiling_z):
    N, C, _ = src_bon.shape
    device = src_bon.device
    theta = ((torch.arange(W).expand(N, C, W).to(device) / W)) * 2 * np.pi #(N, 2, 1024)
    
    z_bons = torch.cat([ceiling_z, torch.zeros_like(ceiling_z) - 1.], dim=-1)
    src_phi = (0.5 - (src_bon / H)) * np.pi # (N, 2, 1024)
    src_dist = z_bons[:, :, None] / torch.tan(src_phi) # (N, 2, 1024)
    src_x = src_dist * torch.sin(theta) # (N, 2, 1024)
    src_y = src_dist * torch.cos(theta + np.pi) # (N, 2, 1024)
    src_local_2d = torch.stack([src_x, src_y], axis=-1).view(N, -1, 2) # (N, 2*1024, 2)
    
    return src_local_2d

def compute_global(src_bon, src_transformer, H, W, ceiling_z):
    src_local_2d = compute_local(src_bon, H, W, ceiling_z) # (N, 2*1024, 2)
    src_global_2d = src_transformer.to_global(src_local_2d) # (N, 2*1024, 2)
    
    return src_global_2d


