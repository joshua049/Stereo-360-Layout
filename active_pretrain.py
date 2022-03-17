import os
import argparse
import numpy as np
import json
from tqdm import trange
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET, DoorNet
from dataset import PanoCorBonDataset, ZillowIndoorDataset
from misc.utils import group_weight, adjust_learning_rate, save_model, load_trained_model
from inference import inference
from eval_general import test_general

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

def feed_forward(net, x, y_bon, y_cor):
    x = x.to(device)
    y_bon = y_bon.to(device)
    y_cor = y_cor.to(device)
    losses = {}

    # y_bon_, y_cor_ = net(x)
    y_bon_= net(x)
    losses['bon'] = F.l1_loss(y_bon_, y_bon)
    # losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor)    
    # losses['total'] = losses['bon'] + losses['cor']
    losses['total'] = losses['bon']

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
    parser.add_argument('--no_train', action='store_true',
                        help='evaluation only')
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
    parser.add_argument('--num_mp3d', default=1650, type=int,
                        help='number of mp3d_layout data')
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

    # doornet = DoorNet()
    # doornet.load_state_dict(torch.load('./ckpt/door_sub/best.pth')['state_dict'])
    # doornet.eval()
    doornet = None

    # Create dataloader
    dataset_train = PanoCorBonDataset(
    # dataset_train = ZillowIndoorDataset(
        root_dir=args.train_root_dir,
        subject='train',
        flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
        stretch=not args.no_pano_stretch, end=args.num_mp3d)
    loader_train = DataLoader(dataset_train, args.batch_size_train,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=lambda x: np.random.seed())
    if args.valid_root_dir:
        dataset_valid = PanoCorBonDataset(
        # dataset_valid = ZillowIndoorDataset(
            root_dir=args.valid_root_dir, subject='val', return_cor=True,
            flip=False, rotate=False, gamma=False,
            stretch=False)

    print('Sup:', len(dataset_train))
    # assert False

    # Create model
    if args.pth is not None:
        print('Finetune model is given.')
        print('Ignore --backbone and --no_rnn')
        net = load_trained_model(HorizonNet, args.pth).to(device)

        finetune_linear_only = False
        if finetune_linear_only:
            print('Finetune Linear Only!')
            for p in net.parameters():
                p.requires_grad = False
            
            for p in net.linear.parameters():
                p.requires_grad = True
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
    args.warmup_iters = args.warmup_epochs * len(loader_train)
    args.max_iters = args.epochs * len(loader_train)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.cur_iter = 0
    args.best_valid_score = 0

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):

        # Train phase
        if not args.no_train:
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
                # Set learning rate
                # break
                adjust_learning_rate(optimizer, args)

                args.cur_iter += 1
                # print('*'*30, args.cur_iter, '*'*30)
                x, y_bon, y_cor = next(iterator_train)

                losses = feed_forward(net, x, y_bon, y_cor)
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
        if args.valid_root_dir:
            valid_loss = {}
            meters = {n_corner: AverageMeter() for n_corner in ['4', '6', '8', '10+', 'odd']}
            excepts = 0
            for jth in trange(len(dataset_valid),
                            desc='Valid ep%d' % ith_epoch, position=2):
                x, y_bon, y_cor, gt_cor_id = dataset_valid[jth]
                x, y_bon, y_cor = x[None], y_bon[None], y_cor[None]
                with torch.no_grad():
                    losses = feed_forward(net, x, y_bon, y_cor)

                    # True eval result instead of training objective
                    true_eval = dict([
                        (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
                        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
                    ])

                    
                    # dt_cor_id = inference(net, doornet, x, device, force_cuboid=False)[0]
                    # dt_cor_id[:, 0] *= 1024
                    # dt_cor_id[:, 1] *= 512
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

                    # print(losses['3DIoU'])                   

                for n_corner in ['4', '6', '8', '10+', 'odd']:
                    meters[n_corner].update(sum(true_eval[n_corner]['3DIoU']), len(true_eval[n_corner]['3DIoU']))

                for k, v in losses.items():
                    try:                        
                        valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)
                    except ValueError:                        
                        valid_loss[k] = valid_loss.get(k, 0)
            print('Num of Exceptions:', excepts)
            for n_corner in ['4', '6', '8', '10+', 'odd']:
                print(f'{n_corner} Corners:', meters[n_corner].avg)

            for k, v in valid_loss.items():
                k = 'valid/%s' % k
                tb_writer.add_scalar(k, v / len(dataset_valid), ith_epoch)

            # Save best validation loss model
            with open('loss.json', 'w') as f: json.dump(valid_loss, f, indent=4)
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
        # if ith_epoch % args.save_every == 0:
        #     save_model(net,
        #                os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch),
        #                args)
