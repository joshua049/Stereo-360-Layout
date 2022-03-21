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

# from model_ori import HorizonNet
from model import HorizonNet, DoorNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils
from eval_general import test_general




def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def inference(net, x, device, visualize=False):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    y_bon_ = net(x.to(device)).detach().cpu()

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
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Post-processing realted
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--min_v', default=None, type=float)
    parser.add_argument('--force_cuboid', action='store_true')
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
    # net = HorizonNet('resnet50', True).to(device)
    net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net.eval()

    doornet = DoorNet()
    doornet.load_state_dict(torch.load('./ckpt/door_sub/best.pth')['state_dict'])

    gaussian_kernel = get_gaussian_kernel(15, 3)

    # Testing
    losses = dict([
        (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
    ])

    # Inferencing
    with torch.no_grad():
        idx = 0
        for i_path in tqdm(paths, desc='Inferencing'):
            # if idx > 2000:
            #     break

            # zind
            k = os.path.split(i_path)[-1][:-4]
            scene_id = os.path.basename(os.path.abspath(os.path.join(i_path, '../../')))
            img_path = os.path.abspath(os.path.join(i_path, '../../panos', f'{k}.jpg'))

            # mp3d
            k = os.path.split(i_path)[-1][:-4]
            img_path = os.path.abspath(os.path.join(i_path, '../img', f'{k}.png'))


            # Load image
            img_pil = Image.open(img_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # try:
            #     # Inferenceing corners
            #     cor_id, z0, z1, vis_out = inference(net, x, device,
            #                                         args.flip, args.rotate,
            #                                         args.visualize,
            #                                         args.force_cuboid,
            #                                         args.min_v, args.r)
                
            
                
            # except:
            #     pass
            cor_id, z0, z1, vis_out = inference(net, doornet, x, device,
                                                    args.flip, args.rotate,
                                                    args.visualize,
                                                    True,
                                                    args.force_cuboid,
                                                    args.min_v, args.r)

            cor_id[:, 0] *= 1024
            cor_id[:, 1] *= 512

            
            with open(i_path) as f:
                gt_cor_id = np.array([l.split() for l in f], np.float32)
            test_general(cor_id, gt_cor_id, 1024, 512, losses)

            last_score = losses['overall']['3DIoU'][-1]
            # print(cor_id, gt_cor_id)
            # print(last_score)
            
            # Output resultrm 
            with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
                json.dump({
                    'z0': float(z0),
                    'z1': float(z1),
                    'uv': [[float(u), float(v)] for u, v in cor_id],
                }, f)
            idx += 1
            if last_score < 0.5 and vis_out is not None:
                vis_path = os.path.join(args.output_dir, f'{scene_id}_{k}.raw.png')
                # print(vis_path)
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                    .resize((vw//2, vh//2), Image.LANCZOS)\
                    .save(vis_path)
        for k, result in losses.items():
            iou2d = np.array(result['2DIoU'])
            iou3d = np.array(result['3DIoU'])
            rmse = np.array(result['rmse'])
            delta_1 = np.array(result['delta_1'])
            if len(iou2d) == 0:
                continue
            print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
            print('    2DIoU  : %.2f' % (iou2d.mean() * 100))
            print('    3DIoU  : %.2f' % (iou3d.mean() * 100))
            print('    RMSE   : %.2f' % (rmse.mean()))
            print('    delta^1: %.2f' % (delta_1.mean()))