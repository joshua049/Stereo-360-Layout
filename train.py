import os
import sys
import argparse
import pdb
import numpy as np
import json
from tqdm import trange
from copy import deepcopy
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.loss import chamfer_distance

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET
from dataset import PanoCorBonDataset, ZInD_SupSet, ZInD_UnSupSet
from misc.utils import group_weight, adjust_learning_rate, save_model, load_trained_model
from inference import inference
from eval_general import test_general
from transformations_torch import *
from DLVR import warp, compute_local, compute_global, inference_ceiling, generate_mask

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        if self.count > 0:
            self.avg = self.sum / self.count
        else: 
            self.avg = 0

def nan_to_num(x):
    nan_map = x.isnan()
    res = nan_map.float() #(N, 2, 1024)
    res[:, 0, :] *= -0.478
    res[:, 1, :] *= 0.425

    x[nan_map] = 0.

    return x + res    

def clip_bon(bon, H):
    # bon.shape = (N, 2, 1024)
    margin = 15
    ceil = torch.clamp(bon[:, 0, :], 2, H/2 - margin)
    floor = torch.clamp(bon[:, 1, :], H/2 + margin, H-2)

    return torch.stack((ceil, floor), dim=1)


def sup_feed_forward(net, x, y_bon, y_cor):
    x = x.to(device)
    y_bon = y_bon.to(device)
    y_cor = y_cor.to(device)
    losses = {}

    y_bon_= net(x)
    losses['bon'] = F.l1_loss(y_bon_, y_bon)
    losses['total'] = losses['bon']

    return losses

def unsup_feed_forward(args, net, params, single=False, epoch=0):

    src_img, src_rotation_matrix, src_scale, src_translation, target_img, target_rotation_matrix, target_scale, target_translation, stretched_src_img, stretched_target_img, stretch_k, ceiling_height = params 
    src_img = src_img.to(device)
    src_rotation_matrix = src_rotation_matrix.to(device)
    src_scale = src_scale.to(device)
    src_translation = src_translation.to(device)
    target_img = target_img.to(device)
    target_rotation_matrix = target_rotation_matrix.to(device)
    target_scale = target_scale.to(device)
    target_translation = target_translation.to(device)
    stretched_target_img = stretched_target_img.to(device)
    stretch_k = stretch_k.to(device)
    ceiling_height = ceiling_height.to(device)

    N, C, H, W = src_img.shape
    src_transformer = Transformation2D(rotation_matrix=src_rotation_matrix, scale=src_scale[:, :, None], translation=src_translation)
    target_transformer = Transformation2D(rotation_matrix=target_rotation_matrix, scale=target_scale[:, :, None], translation=target_translation)
    losses = {'main': None, 'aux': {}} 

    y_bon_ori = net(target_img)
    y_bon = (y_bon_ori / 2 + 0.5) * H - 0.5
    y_bon = clip_bon(y_bon, H)
         
    if args.no_ceiling_ann:
        ceiling_z = guess_ceiling(y_bon, H, W).reshape(ceiling_height.shape)
    else:
        ceiling_z = ceiling_height - 1.

    grid = warp(src_transformer, target_transformer, y_bon, H, W, ceiling_z).to(device)
    warp_img = F.grid_sample(src_img, grid)

    if args.no_validity:
        valid_map = torch.ones(N, C, H, W)
    else:
        src_valid_map = generate_mask(src_img.shape, 45).to(device)
        target_valid_map = F.grid_sample(src_valid_map, grid)
        valid_map = torch.clip(src_valid_map * target_valid_map, min=0., max=1.).detach()

    losses['main'] = (F.mse_loss(warp_img, target_img, reduction='none') * valid_map).mean()

    target_local_2d = compute_local(y_bon, H, W, ceiling_z)
    target_global_2d = compute_global(y_bon, target_transformer, H, W, ceiling_z)   

    if not args.no_cycle:
        pseudo_y_bon_ori = net(warp_img.detach())
        pseudo_y_bon = (pseudo_y_bon_ori / 2 + 0.5) * H - 0.5
        pseudo_y_bon = clip_bon(pseudo_y_bon, H)    

        losses['aux']['cycle'] = F.mse_loss(pseudo_y_bon_ori, y_bon_ori.detach())    
    
    if not args.no_srctgt:
        y_bon_src = nan_to_num(net(src_img))
        y_bon_src = (y_bon_src / 2 + 0.5) * H - 0.5
        y_bon_src = clip_bon(y_bon_src, H)
        src_global_2d = compute_global(y_bon_src, src_transformer, H, W, ceiling_z)   
        losses['aux']['src_tgt'] = chamfer_distance(src_global_2d, target_global_2d)[0]

    if not args.no_manhattan:
        target_global_2d_translate = target_global_2d - target_translation
        kernel_size = 15
        unfold = nn.Unfold(kernel_size=(1, kernel_size))
        windows = unfold(target_global_2d_translate.reshape(N, 2, W, 2).permute(0, 1, 3, 2)).reshape(N, 2, 2, -1, kernel_size) #(N, C, XY, L, K)
        windows_mean = windows.median(dim=-1, keepdim=True).values
        windows_slope = windows / windows.norm(dim=2, keepdim=True)   
        losses['aux']['manhattan'] = torch.abs((windows - windows_mean) * torch.flip(windows_slope, [2])).mean(dim=4).min(dim=2).values.mean()  
    
    if not args.no_ceilfloor:
        ceil = target_global_2d[:, :1024, :]
        floor = target_global_2d[:, 1024:, :]
        losses['aux']['ceil_floor'] = F.mse_loss(ceil, floor)

    if not args.no_stretchaug:
        target_local_stretch = target_local_2d * stretch_k[:, None, :]    
        stretched_y_bon_ = net(stretched_target_img)
        stretched_y_bon_ = (stretched_y_bon_ / 2 + 0.5) * H - 0.5  
        stretched_y_bon_ = clip_bon(stretched_y_bon_, H)
        target_local_stretch_ = compute_local(stretched_y_bon_, H, W, ceiling_z)
        losses['aux']['stretch'] = chamfer_distance(target_local_stretch.detach(), target_local_stretch_)[0]  

    losses['total'] = losses['main'] + 0.1 * sum(losses['aux'].values())

    return losses



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', required=True,
                        help='experiment id to name checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--logs', default='./logs',
                        help='folder to logging')
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint.'
                             '(finetuning)')
    parser.add_argument('--log_to_file', action='store_true',
                        help='redirect stdout to logs/{id}.log')
    parser.add_argument('--eval_only', action='store_true',
                        help='evaluation only')
    # Model related
    parser.add_argument('--backbone', default='resnet50',
                        choices=ENCODER_RESNET + ENCODER_DENSENET,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true',
                        help='whether to remove rnn or not')
    # Dataset related arguments
    parser.add_argument('--sup_root_dir', default=None,
                        help='root directory to supervised training dataset.')
    parser.add_argument('--unsup_root_dir', default=None,
                        help='root directory to unsupervised training dataset.')
    parser.add_argument('--valid_root_dir', required=True,
                        help='root directory to validation dataset.'
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--no_flip', action='store_true',
                        help='disable left-right flip augmentation')
    parser.add_argument('--no_rotate', action='store_true',
                        help='disable horizontal rotate augmentation')
    parser.add_argument('--no_gamma', action='store_true',
                        help='disable gamma augmentation')
    parser.add_argument('--no_pano_stretch', action='store_true',
                        help='disable pano stretch')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='numbers of workers for dataloaders')
    parser.add_argument('--num_mp3d', default=None, type=int,
                        help='number of mp3d_layout data')
    # DLVR related arguments
    parser.add_argument('--no_validity', action='store_true',
                        help='remove validity mask')
    parser.add_argument('--no_cycle', action='store_true',
                        help='remove cycle consistency')
    parser.add_argument('--no_srctgt', action='store_true',
                        help='remove src-tgt consistency')
    parser.add_argument('--no_manhattan', action='store_true',
                        help='remove manhattan alighment')
    parser.add_argument('--no_ceilfloor', action='store_true',
                        help='remove ceiling-floor consistency')
    parser.add_argument('--no_stretchaug', action='store_true',
                        help='remove stretch augmentation consistency')
    parser.add_argument('--no_ceiling_ann', action='store_true',
                        help='render the image without ceiling annotations')
    # optimization related arguments
    parser.add_argument('--freeze_earlier_blocks', default=-1, type=int)
    parser.add_argument('--batch_size_train', default=4, type=int,
                        help='training mini-batch size')
    parser.add_argument('--batch_size_valid', default=2, type=int,
                        help='validation mini-batch size')
    parser.add_argument('--sup_epochs', default=0, type=int,
                        help='epochs for supervised training')
    parser.add_argument('--unsup_epochs', default=0, type=int,
                        help='epochs for unsupervised training')
    parser.add_argument('--optim', default='Adam',
                        help='optimizer to use. only support SGD and Adam')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--warmup_lr', default=1e-6, type=float,
                        help='starting learning rate for warm up')
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='numbers of warmup epochs')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='factor for L2 regularization')
    parser.add_argument('--bn_momentum', type=float)
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--seed', default=594277, type=int,
                        help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='iterations frequency to display')
    parser.add_argument('--save_every', type=int, default=25,
                        help='epochs frequency to save state_dict')
    args = parser.parse_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)


    # Create dataloader
    dataset_name = os.path.basename(args.valid_root_dir)
    if dataset_name == 'zind':
        if args.eval_only:
            dataset_valid = ZInD_SupSet(
            root_dir=args.valid_root_dir, subject='test', 
            flip=False, rotate=False, gamma=False, return_cor=True,
            stretch=False)
        else:
            dataset_valid = ZInD_SupSet(
            root_dir=args.valid_root_dir, subject='val', 
            flip=False, rotate=False, gamma=False, return_cor=True,
            stretch=False)
    else:
        dataset_valid = PanoCorBonDataset(
        root_dir=args.valid_root_dir, 
        flip=False, rotate=False, gamma=False, return_cor=True,
        stretch=False)

    if args.eval_only:
        args.sup_epochs = 1

    else:
        if args.sup_root_dir is not None: 
            dataset_name = os.path.basename(args.sup_root_dir) 
            if dataset_name == 'zind':    
                sup_dataset_train = ZInD_SupSet(
                    root_dir=args.sup_root_dir,
                    subject='train',
                    flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
                    stretch=not args.no_pano_stretch)
            else:
                sup_dataset_train = PanoCorBonDataset(
                    root_dir=args.sup_root_dir,
                    flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
                    stretch=not args.no_pano_stretch,
                    sample_num=None)

            sup_loader_train = DataLoader(sup_dataset_train, args.batch_size_train * 4,
                                    shuffle=True, drop_last=True,
                                    num_workers=args.num_workers,
                                    pin_memory=not args.no_cuda,
                                    worker_init_fn=lambda x: np.random.seed())
            if args.sup_epochs == 0:
                args.sup_epochs = 100

        if args.unsup_root_dir is not None:
            unsup_dataset_train = ZInD_UnSupSet(
                root_dir=args.unsup_root_dir,
                subject='train',
                flip=False, rotate=False, gamma=False,
                stretch=True,
                max_stretch=1.5)
            unsup_loader_train = DataLoader(unsup_dataset_train, args.batch_size_train,
                                    shuffle=True, drop_last=True,
                                    num_workers=args.num_workers,
                                    pin_memory=not args.no_cuda,
                                    worker_init_fn=lambda x: np.random.seed())
            if args.unsup_epochs == 0:
                args.unsup_epochs = 20

        print('Sup:', len(sup_dataset_train))
        print('UnSup:', len(unsup_dataset_train))

    print('Valid:', len(dataset_valid))

    # assert False


    # Create model
    if args.pth is not None:
        print('Finetune model is given.')
        print('Ignore --backbone and --no_rnn')
        net = load_trained_model(HorizonNet, args.pth).to(device)
    else:
        net = HorizonNet(args.backbone, not args.no_rnn).to(device)
    
    assert -1 <= args.freeze_earlier_blocks and args.freeze_earlier_blocks <= 4
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            print('Freeze block%d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

    if args.bn_momentum:
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = args.bn_momentum

    # Create optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    # Create tensorboard for monitoring training
    tb_path = os.path.join(args.logs, args.id)
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    # Init variable
    args.epochs = args.sup_epochs + args.unsup_epochs
    args.warmup_iters = args.warmup_epochs * (len(sup_loader_train))
    args.max_iters = args.epochs * (len(sup_loader_train))
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    
    args.cur_iter = 0
    args.best_valid_score = -1

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):
        # Train phase
        if not args.eval_only:
            net.train()
            if args.freeze_earlier_blocks != -1:
                b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
                blocks = [b0, b1, b2, b3, b4]
                for i in range(args.freeze_earlier_blocks + 1):
                    for m in blocks[i]:
                        m.eval()
            
            if ith_epoch < args.unsup_epochs:
                iterator_train = iter(unsup_loader_train)
                for _ in trange(len(unsup_loader_train),
                                desc='Unsup Train ep%s' % ith_epoch, position=1):
                    # Set learning ratetmux
                    adjust_learning_rate(optimizer, args)

                    args.cur_iter += 1
                    params = next(iterator_train)
                    losses = unsup_feed_forward(args, net, params, epoch=ith_epoch)
                        
                    for k, v in losses.items():
                        if isinstance(v, dict):
                            for k_, v_ in v.items():
                                k = 'train/%s' % k
                                tb_writer.add_scalar(k, v_.item(), args.cur_iter)
                        else:
                            k = 'train/%s' % k
                            tb_writer.add_scalar(k, v.item(), args.cur_iter)
                    tb_writer.add_scalar('train/lr', args.running_lr, args.cur_iter)

                    loss = losses['total']
                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type='inf')
                    optimizer.step()
            
            if args.sup_root_dir is not None and ith_epoch > args.epochs // 2:
                iterator_train = iter(sup_loader_train)
                for _ in trange(len(sup_loader_train),
                                desc='Sup Train ep%s' % ith_epoch, position=1):
                    # Set learning rate
                    adjust_learning_rate(optimizer, args)

                    args.cur_iter += 1
                    # print('*'*30, args.cur_iter, '*'*30)
                    x, y_bon, y_cor = next(iterator_train)

                    losses = sup_feed_forward(net, x, y_bon, y_cor)
                    for k, v in losses.items():
                        k = 'train/%s' % k
                        tb_writer.add_scalar(k, v.item(), args.cur_iter)
                    tb_writer.add_scalar('train/lr', args.running_lr, args.cur_iter)
                    loss = losses['total']

                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type='inf')
                    optimizer.step()

                    # break
            
        
        # Valid phase
        net.eval()
        valid_loss = {}
        meters = {metric: AverageMeter() for metric in ['2DIoU', '3DIoU', 'rmse', 'delta_1']} 
        excepts = 0
        for jth in trange(len(dataset_valid),
                        desc='Valid ep%d' % ith_epoch, position=2):
            x, y_bon, y_cor, gt_cor_id = dataset_valid[jth]
            x, y_bon, y_cor = x[None], y_bon[None], y_cor[None]
            with torch.no_grad():
                losses = sup_feed_forward(net, x, y_bon, y_cor)

                # True eval result instead of training objective
                true_eval = dict([
                    (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
                    for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
                ])

                try:
                    dt_cor_id = inference(net, doornet, x, device, force_cuboid=False, force_raw=True)[0]
                    dt_cor_id[:, 0] *= 1024
                    dt_cor_id[:, 1] *= 512
                except Exception as e:
                    print(e)
                    excepts += 1
                    dt_cor_id = np.array([
                        [k//2 * 1024, 256 - ((k%2)*2 - 1) * 120]
                        for k in range(8)
                    ])
                test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
                losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
                losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
                losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
                losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']]) 

            for metric in ['2DIoU', '3DIoU', 'rmse', 'delta_1']:
                meters[metric].update(sum(true_eval['overall'][metric]), len(true_eval['overall'][metric]))
            
            for k, v in losses.items():
                try:                        
                    valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)
                except ValueError:                        
                    valid_loss[k] = valid_loss.get(k, 0)
        print('Num of Exceptions:', excepts)
        for metric in ['2DIoU', '3DIoU', 'rmse', 'delta_1']:
            print(f'{metric}:', meters[metric].avg)
        
        for k, v in valid_loss.items():
            k = 'valid/%s' % k
            tb_writer.add_scalar(k, v / len(dataset_valid), ith_epoch)

        # Save best validation loss model
        now_valid_score = valid_loss['3DIoU'] / len(dataset_valid)
        print('Ep%3d %.4f vs. Best %.4f' % (ith_epoch, now_valid_score, args.best_valid_score))
        if now_valid_score >= args.best_valid_score:
            args.best_valid_score = now_valid_score
            
            save_model(net,
                        os.path.join(args.ckpt, args.id, f'best_valid.pth'),
                        args)
        save_model(net,
                    os.path.join(args.ckpt, args.id, 'last.pth'),
                    args)

        # Periodically save model
        if ith_epoch % args.save_every == 0:
            save_model(net,
                       os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch),
                       args)
