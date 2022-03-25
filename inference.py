import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from scipy import signal as sg
from shapely.geometry import Polygon


import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HorizonNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils
from eval_general import test_general

def inference(net, x, device, visualize=False):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward 
    y_bon_ = net(x.to(device)).detach().cpu()
    y_cor_ = torch.zeros(1, 1, W)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    y_bon_ = (y_bon_[0] / 2 + 0.5) * H - 0.5
    cor = np.vstack([np.arange(1024), y_bon_]).T

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, vis_out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', required=True,
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--visualize', action='store_true')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Loaded trained model
    net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net.eval()

    # Inferencing
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            # Load image
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # Inferenceing corners
            cor_id, vis_out = inference(net=net, x=x, device=device, visualize=args.visualize)

            # Output result
            with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
                json.dump({
                    'uv': [[float(u), float(v)] for u, v in cor_id],
                }, f)

            if vis_out is not None:
                vis_path = os.path.join(args.output_dir, k + '.raw.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)