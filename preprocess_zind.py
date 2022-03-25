import os
import shutil
import glob
import json
import argparse
import numpy as np
from transformations_np import TransformationSpherical, Transformation3D
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True,
                        help='Path to original ZInD')
    args = parser.parse_args()

    label_fnames = glob.glob(os.path.join(args.src, '*', 'zind_data.json'))
    simple_label_file = os.path.join(args.src, 'room_shape_simplicity_labels.json')
    with open(simple_label_file, 'r') as f: simple_data = json.load(f)

    self_simple_list = []
    official_simple_list = []
    num = 0
    excepts = 0

    for path in tqdm(label_fnames):
        with open(path, 'r') as f: data = json.load(f)
        dirname = os.path.dirname(path)
        label_dir = os.path.join(dirname, 'label')
        single_dir = os.path.join(label_dir, 'singled')
        pair_dir = os.path.join(label_dir, 'paired')
        
        if os.path.isdir(label_dir):
            shutil.rmtree(label_dir)

        os.mkdir(label_dir)        
        scene_id = os.path.basename(dirname)    
        
        merger = data['merger']
        for floor_id, floor_data in merger.items():
            for complete_room_id, complete_room_data in floor_data.items(): 
                for partial_room_id, partial_room_data in complete_room_data.items():
                    for pano_id, pano_data in partial_room_data.items():
                        subject = f'{scene_id}_{floor_id}_{complete_room_id}_{partial_room_id}_{pano_id}' 
                        if simple_data[subject] and 'layout_visible' in pano_data:  
                            vertices = np.array(pano_data['layout_visible']['vertices'])
                            ceiling_height = pano_data['ceiling_height']
                            camera_height = pano_data['camera_height']

                            xyz_floor, xyz_ceil = Transformation3D(ceiling_height, camera_height).to_3d(vertices)
                            
                            xyz_floor = xyz_floor[::-1]
                            xyz_ceil = xyz_ceil[::-1]                        
                            
                            xyzs = np.empty((xyz_floor.shape[0]*2, xyz_floor.shape[1]), dtype=xyz_floor.dtype)
                            xyzs[0::2] = xyz_ceil
                            xyzs[1::2] = xyz_floor

                            cor = TransformationSpherical.cartesian_to_pixel(xyzs, 1024).astype(int) 
                             
                            cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)                         

                            file_prefix = f'{floor_id}_{partial_room_id}_{pano_id}'
                            if pano_data['is_primary']:
                                file_prefix = f'{file_prefix}_primary'

                            txt_filename = os.path.join(label_dir, f'{file_prefix}.txt')
                            np.savetxt(txt_filename, cor, delimiter=' ', fmt='%d')
                            json_filename = os.path.join(label_dir, f'{file_prefix}.json')
                            with open(json_filename, 'w') as f: json.dump(pano_data, f)
                            num += 1
                            
    print(f'Number of Data: {num}')
    
