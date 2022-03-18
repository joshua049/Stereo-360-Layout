from genericpath import isdir
from logging import root
import os
from posixpath import split
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
import glob
import json
import itertools
import pandas as pd

import torch
import torch.utils.data as data

from misc import panostretch
from transformations_torch import *

class PanoCorBonDataset(data.Dataset):
    '''
    See README.md for how to prepare the dataset.
    '''

    def __init__(self, root_dir, subject=None,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 p_base=0.96, max_stretch=2.0,
                 normcor=False, return_cor=False, return_path=False, 
                 start=None, end=None, labeled=True, teach=False):
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        
        if end in [50, 100, 200, 400]:
            # split_data = pd.read_csv('matterport_train_splits.csv')
            split_data = pd.read_csv('active_select_splits.csv')            
            self.img_fnames = sorted([
                f'{fname}.png' for fname in split_data['fileid'][split_data[str(end)]==int(labeled)]
            ])
        else:
            all_img_fnames = sorted([
                fname for fname in os.listdir(self.img_dir)
                if fname.endswith('.jpg') or fname.endswith('.png')
            ])
            self.img_fnames = all_img_fnames[start:end]

        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_cor = return_cor
        self.return_path = return_path
        self.teach = teach

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)),\
                '%s not found' % os.path.join(self.cor_dir, fname)

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.             
        H, W = img.shape[:2]

        # Read ground truth corners
        with open(os.path.join(self.cor_dir,
                               self.txt_fnames[idx])) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            # Detect occlusion
            occlusion = find_occlusion(cor[::2].copy()).repeat(2)
            assert (np.abs(cor[0::2, 0] - cor[1::2, 0]) > W/100).sum() == 0, img_path
            assert (cor[0::2, 1] > cor[1::2, 1]).sum() == 0, img_path
       
        # Stretch augmentation
        if self.stretch:
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, cor = panostretch.pano_stretch(img, cor, kx, ky)

        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, H, W)

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            bon = np.flip(bon, axis=1)
            cor[:, 0] = img.shape[1] - 1 - cor[:, 0]

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            bon = np.roll(bon, dx, axis=1)
            cor[:, 0] = (cor[:, 0] + dx) % img.shape[1]

        original_img = img.copy()           
        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p

        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]       
        
        dist_o = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1))
                    #    p=1)
        dist_r = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) + img.shape[1])
                    #    p=1)
        dist_l = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) - img.shape[1])
                    #    p=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        nearest_dist = dist.min(0)
        y_cor = (self.p_base ** nearest_dist).reshape(1, -1)

        # Convert all data to tensor
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        original_x = torch.FloatTensor(original_img.transpose([2, 0, 1]).copy())
        bon = torch.FloatTensor(bon.copy())
        y_cor = torch.FloatTensor(y_cor.copy())

        # Check whether additional output are requested
        out_lst = [x, bon, y_cor]
        if self.teach:
            out_lst.append(original_x)
        if self.return_cor:
            out_lst.append(cor)
        if self.return_path:
            out_lst.append(img_path)

        return out_lst

class ZInD_SupSet(data.Dataset):

    def __init__(self, root_dir, subject,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 p_base=0.96, max_stretch=2.0,
                 normcor=False, return_cor=False, return_path=False, start=None, end=None):
        assert subject in ['train', 'val', 'test'], root_dir
        with open(os.path.join(root_dir, 'zind_partition.json')) as f: split_data = json.load(f)
        scene_ids = split_data[subject][start:end]
        self.label_fnames = []
        for scene_id in scene_ids:
            scene_dir = os.path.join(root_dir, scene_id)
            self.label_fnames += glob.glob(os.path.join(scene_dir, 'label', '*.txt'))
        
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_cor = return_cor
        self.return_path = return_path

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.label_fnames:
            assert os.path.isfile(fname),\
                '%s not found' % fname

    def __len__(self):
        return len(self.label_fnames)

    def __getitem__(self, idx):
        label_path = self.label_fnames[idx]
        dirname, basename = os.path.split(label_path)
        basename = basename.split('.')[0]
        scene_dir = os.path.dirname(dirname)
        img_path = os.path.join(scene_dir, 'panos', f'{basename}.jpg')
        img = np.array(Image.open(img_path).resize((1024, 512)), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]

        cor = np.loadtxt(label_path, delimiter=' ', dtype=int)

        # Corner with minimum x should at the beginning
        cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

        # Detect occlusion
        occlusion = find_occlusion(cor[::2].copy()).repeat(2)
        assert (np.abs(cor[0::2, 0] - cor[1::2, 0]) > W/100).sum() == 0, img_path
        assert (cor[0::2, 1] > cor[1::2, 1]).sum() == 0, img_path

        # Stretch augmentation
        if self.stretch:
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, cor = panostretch.pano_stretch(img, cor, kx, ky)

        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, H, W)

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            bon = np.flip(bon, axis=1)
            cor[:, 0] = img.shape[1] - 1 - cor[:, 0]

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            bon = np.roll(bon, dx, axis=1)
            cor[:, 0] = (cor[:, 0] + dx) % img.shape[1]

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p

        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]      
        dist_o = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1))
                    #    p=1)
        dist_r = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) + img.shape[1])
                    #    p=1)
        dist_l = cdist(corx.reshape(-1, 1),
                       np.arange(img.shape[1]).reshape(-1, 1) - img.shape[1])
                    #    p=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        nearest_dist = dist.min(0)
        y_cor = (self.p_base ** nearest_dist).reshape(1, -1)

        # Convert all data to tensor
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        bon = torch.FloatTensor(bon.copy())
        y_cor = torch.FloatTensor(y_cor.copy())
        # print(bon.shape, y_cor.shape)

        # Check whether additional output are requested
        out_lst = [x, bon, y_cor]
        if self.return_cor:
            out_lst.append(cor)
        if self.return_path:
            out_lst.append(img_path)

        return out_lst

class ZInD_UnSupSet(data.Dataset):
    def __init__(self, root_dir, subject,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 p_base=0.96, max_stretch=2.0,
                 normcor=False, return_cor=False, return_path=False, start=None, end=None):

        assert subject in ['train', 'val', 'test'], root_dir
        with open(os.path.join(root_dir, 'zind_partition.json')) as f: split_data = json.load(f)
        scene_ids = split_data[subject][start:end]
        all_jsons = []
        for scene_id in scene_ids:
            scene_dir = os.path.join(root_dir, scene_id)
            all_jsons += glob.glob(os.path.join(scene_dir, 'label', '*.json'))
            
        all_scenes = {}
        for filename in all_jsons:
            partial_room_id, pano_id = filename.split('_pano_')
            if partial_room_id in all_scenes:
                if pano_id.endswith('_primary'):
                    all_scenes[partial_room_id].insert(0, pano_id)
                else:
                    all_scenes[partial_room_id].append(pano_id)
            else:
                all_scenes[partial_room_id] = [pano_id]
        
        self.scenes = []
        for k, v in all_scenes.items():
            if len(v) >= 2:
                file_pair = (f'{k}_pano_{v[0]}', f'{k}_pano_{v[1]}')
                self.scenes.append(file_pair)                
                
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_cor = return_cor
        self.return_path = return_path       
        
        # self._check_dataset() 

    def _check_dataset(self):
        for src_path, target_path in self.scenes:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)),\
                '%s not found' % os.path.join(self.cor_dir, fname)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        src_path, target_path = self.scenes[idx]
        
        label_path = src_path
        dirname, basename = os.path.split(label_path)
        basename = basename.split('.')[0]
        if basename.endswith('_primary'):
            basename = basename.split('_primary')[0]
        scene_dir = os.path.dirname(dirname)
        img_path = os.path.join(scene_dir, 'panos', f'{basename}.jpg')
        with open(label_path) as f: src_ann = json.load(f)
        src_img = np.array(Image.open(img_path).resize((1024, 512)), np.float32)[..., :3] / 255.
        H, W = src_img.shape[:2]

        
        label_path = target_path
        dirname, basename = os.path.split(label_path)
        basename = basename.split('.')[0]
        if basename.endswith('_primary'):
            basename = basename.split('_primary')[0]
        scene_dir = os.path.dirname(dirname)
        img_path = os.path.join(scene_dir, 'panos', f'{basename}.jpg')
        with open(label_path) as f: target_ann = json.load(f)
        target_img = np.array(Image.open(img_path).resize((1024, 512)), np.float32)[..., :3] / 255.
        

        src_transformer = Transformation2D.from_zind_data(src_ann['floor_plan_transformation'])
        target_transformer = Transformation2D.from_zind_data(target_ann['floor_plan_transformation'])
        
        H, W = src_img.shape[:2]

        kx, ky = 1, 1
        # Stretch augmentation
        if self.stretch:
            xmin, ymin, xmax, ymax = cor2xybound(target_cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            kx = max(1 / kx, min(0.5 / xmin, 1.0))
            ky = max(1 / ky, min(0.5 / ymin, 1.0))
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))           

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p

        stretched_src_img, _ = panostretch.pano_stretch(src_img, None, kx, ky)
        stretched_target_img, _ = panostretch.pano_stretch(target_img, None, kx, ky)

        # Convert all data to tensor
        src_img = torch.FloatTensor(src_img.transpose([2, 0, 1]).copy())
        target_img = torch.FloatTensor(target_img.transpose([2, 0, 1]).copy())
        stretched_src_img = torch.FloatTensor(stretched_src_img.transpose([2, 0, 1]).copy())
        stretched_target_img = torch.FloatTensor(stretched_target_img.transpose([2, 0, 1]).copy())
        ceiling_height = torch.FloatTensor([src_ann['ceiling_height']])
        stretch_k = torch.FloatTensor([ky, kx])

        out_lst = [ src_img, src_transformer.rotation_matrix, src_transformer.scale, src_transformer.translation, 
                    target_img, target_transformer.rotation_matrix, target_transformer.scale, target_transformer.translation, 
                    stretched_src_img, stretched_target_img, stretch_k, ceiling_height
                ]

        return out_lst    

 
    
def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
        
    bon = np.zeros((2, W))
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)       
    
    # bon = ((bon + 0.5) / H - 0.5) * np.pi
    bon = ((bon + 0.5) / H - 0.5) * 2
    return bon


def infinite_data_generator(data_loader):
    while True:
        for data in data_loader:
            yield data
            

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        if len(other_layout) >= 2:
            other_layout = LineString(other_layout)
            occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
    y_bon = np.clip(((y_bon / 2 + 0.5) * x.shape[0]-1).round().astype(int), 0, x.shape[0]-1)
    # y_bon = np.clip(((y_bon / np.pi + 0.5) * x.shape[0]-1).round().astype(int), 0, x.shape[0]-1)
    # print(y_bon)
    y_cor = y_cor.numpy()

    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy() * 0.5).astype(np.uint8)
    y1 = np.round(y_bon[0]).astype(int)
    y2 = np.round(y_bon[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

    return np.concatenate([gt_cor, img_pad, img_bon], 0)


def visualize_door(x, door_bar):

    gt_door = np.zeros((30, 1024, 3), np.uint8)
    gt_door[:] = door_bar[0][None, :, None] * 255


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/valid/')
    parser.add_argument('--ith', default=0, type=int,
                        help='Pick a data id to visualize.'
                             '-1 for visualize all data')
    parser.add_argument('--flip', action='store_true',
                        help='whether to random flip')
    parser.add_argument('--rotate', action='store_true',
                        help='whether to random horizon rotation')
    parser.add_argument('--gamma', action='store_true',
                        help='whether to random luminance change')
    parser.add_argument('--stretch', action='store_true',
                        help='whether to random pano stretch')
    parser.add_argument('--out_dir', default='sample_dataset_visualization')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    # dataset = PanoCorBonDataset(
    #     root_dir=args.root_dir,
    #     flip=args.flip, rotate=args.rotate, gamma=args.gamma, stretch=args.stretch,
    #     return_path=True)

    dataset = ZillowIndoorDoorDataset(
        root_dir=args.root_dir, subject='val', 
        flip=False, rotate=False, gamma=False,
        stretch=False, return_path=True)

    # Showing some information about dataset
    print('len(dataset): {}'.format(len(dataset)))
    # x, y_bon, y_cor, path = dataset[0]
    x, door_bar = dataset[0]
    print('x', x.size())
    print('door_bar', y_bon.size())
    # print('y_cor', y_cor.size())

    if args.ith >= 0:
        to_visualize = [dataset[args.ith]]
    else:
        to_visualize = dataset

    for x, y_bon, y_cor, path in tqdm(to_visualize):
        print(path)
        fname = os.path.split(path)[-1]
        out = visualize_a_data(x, y_bon, y_cor)
        Image.fromarray(out).save(os.path.join(args.out_dir, fname))
