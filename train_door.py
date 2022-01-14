import os
import argparse
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
from sklearn.metrics import f1_score

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET, DoorNet
from dataset import PanoCorBonDataset, ZillowIndoorDataset, ZillowIndoorPairDataset, ZillowIndoorDoorDataset
from misc.utils import group_weight, adjust_learning_rate, save_model, load_trained_model
from inference import inference
from eval_general import test_general
from transformations_torch import *
from grid import warp_index, compute_global

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
    ceil = torch.clamp(bon[:, 0, :], 0.5, H/2 - 45)
    floor = torch.clamp(bon[:, 1, :], H/2 + 45, H - 0.5)

    return torch.stack((ceil, floor), dim=1)


def feed_forward(net, criterion, params, single=False, epoch=0):

    x, door_bar = params

    if single:
        x = x[None]
        door_bar = door_bar[None]
    
    x = x.to(device)
    door_bar = door_bar.to(device)

    N, _, H, W = x.shape

    predict_door = net(x)

    losses = {} 
    losses['total'] = criterion(predict_door, door_bar)
    losses['score'] = f1_score(torch.where(predict_door > 0.5, 1., 0.).reshape(-1).detach().cpu(), door_bar.reshape(-1).detach().cpu())
    
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
    # Model related
    parser.add_argument('--backbone', default='resnet50',
                        choices=ENCODER_RESNET + ENCODER_DENSENET,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true',
                        help='whether to remove rnn or not')
    # Dataset related arguments
    parser.add_argument('--train_root_dir', default='data/layoutnet_dataset/train',
                        help='root directory to training dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--valid_root_dir', default='data/layoutnet_dataset/valid',
                        help='root directory to validation dataset. '
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
    # optimization related arguments
    parser.add_argument('--freeze_earlier_blocks', default=-1, type=int)
    parser.add_argument('--batch_size_train', default=4, type=int,
                        help='training mini-batch size')
    parser.add_argument('--batch_size_valid', default=2, type=int,
                        help='validation mini-batch size')
    parser.add_argument('--epochs', default=300, type=int,
                        help='epochs to train')
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
    # dataset_train = PanoCorBonDataset(
    dataset_train = ZillowIndoorDoorDataset(
        root_dir=args.train_root_dir,
        subject='val',
        flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
        stretch=not args.no_pano_stretch)
    loader_train = DataLoader(dataset_train, args.batch_size_train,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=lambda x: np.random.seed())
    if args.valid_root_dir:
        # dataset_valid = PanoCorBonDataset(
        dataset_valid = ZillowIndoorDoorDataset(
            root_dir=args.valid_root_dir, subject='test', return_cor=True,
            flip=False, rotate=False, gamma=False,
            stretch=False)

    # Create model
    # if args.pth is not None:
    #     print('Finetune model is given.')
    #     print('Ignore --backbone and --no_rnn')
    #     net = load_trained_model(HorizonNet, args.pth).to(device)
    # else:
    #     net = HorizonNet(args.backbone, not args.no_rnn).to(device)

    # criterion = torch.nn.MSELoss()
    net = DoorNet().to(device)
    criterion = torch.nn.BCELoss()
    
    
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

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7, min_lr=1e-6)
    # Create tensorboard for monitoring training
    tb_path = os.path.join(args.logs, args.id)
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    # Init variable
    args.warmup_iters = args.warmup_epochs * len(loader_train)
    args.max_iters = args.epochs * len(loader_train)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.cur_iter = 0
    args.best_valid_score = -10000

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):

        # Train phase
        net.train()
        if args.freeze_earlier_blocks != -1:
            b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(args.freeze_earlier_blocks + 1):
                for m in blocks[i]:
                    m.eval()
        iterator_train = iter(loader_train)
        for _ in trange(len(loader_train),
                        desc='Train ep%s' % ith_epoch, position=1):
            # Set learning ratetmux
            adjust_learning_rate(optimizer, args)

            args.cur_iter += 1
            # print('*'*30, args.cur_iter, '*'*30)
            params = next(iterator_train)

            losses = feed_forward(net, criterion, params, epoch=ith_epoch)
            for k, v in losses.items():
                k = 'train/%s' % k
                tb_writer.add_scalar(k, v.item(), args.cur_iter)
            tb_writer.add_scalar('train/lr', args.running_lr, args.cur_iter)
            loss = losses['total']

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # for p in net.parameters():
            #     if p.grad is not None:
            #         print(p.grad)

            nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type='inf')
            optimizer.step()

        # Valid phase
        net.eval()
        if args.valid_root_dir:
            valid_loss = {}
            meters = {n_corner: AverageMeter() for n_corner in ['4', '6', '8', '10+', 'odd']}
            excepts = 0
            loss_meter = AverageMeter()
            for jth in trange(len(dataset_valid),
                            desc='Valid ep%d' % ith_epoch, position=2):
                # if jth > 1:
                #     break
                # x, y_bon, y_cor, gt_cor_id = dataset_valid[jth]
                # x, y_bon, y_cor = x[None], y_bon[None], y_cor[None]
                params = dataset_valid[jth]
                with torch.no_grad():
                    losses = feed_forward(net, criterion, params, single=True)
                    loss_meter.update(losses['score'].item(), 1)
                    # True eval result instead of training objective
                    # true_eval = dict([
                    #     (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
                    #     for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
                    # ])

                    # try:
                    #     dt_cor_id = inference(net, x, device, force_cuboid=False, force_raw=True)[0]
                    #     dt_cor_id[:, 0] *= 1024
                    #     dt_cor_id[:, 1] *= 512
                    # except:
                    #     excepts += 1
                    #     dt_cor_id = np.array([
                    #         [k//2 * 1024, 256 - ((k%2)*2 - 1) * 120]
                    #         for k in range(8)
                    #     ])
                    # # dt_cor_id = inference(net, x, device, force_cuboid=False, force_raw=True)[0]
                    # # dt_cor_id[:, 0] *= 1024
                    # # dt_cor_id[:, 1] *= 512

                    # test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
                    # losses = {}
                    # losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
                    # losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
                    # losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
                    # losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])       

                    # loss_meter.update(losses['3DIoU'].mean().item(), 1)             
                    # print(losses['3DIoU'].mean().item())

                # for n_corner in ['4', '6', '8', '10+', 'odd']:
                #     meters[n_corner].update(sum(true_eval[n_corner]['3DIoU']), len(true_eval[n_corner]['3DIoU']))

                # for k, v in losses.items():
                #     try:                        
                #         valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)
                #     except ValueError:                        
                #         valid_loss[k] = valid_loss.get(k, 0)
            # for n_corner in ['4', '6', '8', '10+', 'odd']:
            #     print(f'{n_corner} Corners:', meters[n_corner].avg)

            # for k, v in valid_loss.items():
            #     k = 'valid/%s' % k
            #     tb_writer.add_scalar(k, v / len(dataset_valid), ith_epoch)

            # Save best validation loss model
            # now_valid_score = valid_loss['3DIoU'] / len(dataset_valid)
            now_valid_score = loss_meter.avg
            scheduler.step(now_valid_score)
            print('Ep%3d %.4f vs. Best %.4f' % (ith_epoch, now_valid_score, args.best_valid_score))
            if now_valid_score >= args.best_valid_score:
                args.best_valid_score = now_valid_score
                
                save_model(net,
                           os.path.join(args.ckpt, args.id, f'best_valid_{ith_epoch}.pth'),
                           args)
            save_model(net,
                       os.path.join(args.ckpt, args.id, 'last.pth'),
                       args)

        # Periodically save model
        if ith_epoch % args.save_every == 0:
            save_model(net,
                       os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch),
                       args)
