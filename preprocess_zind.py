import os
import shutil
import glob
import json
import argparse
import numpy as np
from transformations_np import TransformationSpherical, Transformation3D
from tqdm import tqdm

def is_loop_closure_line(width: int, pt1: np.ndarray, pt2: np.ndarray):
    """Check if a given line is a "loop closure line", meaning that it's rendering on the pano texture would
    wrap around the left/right border.
    """
    pt1 = pt1.reshape(1, 3)
    pt2 = pt2.reshape(1, 3)

    pt1_pix = TransformationSpherical.cartesian_to_pixel(pt1, width)
    pt2_pix = TransformationSpherical.cartesian_to_pixel(pt2, width)

    mid_pt = 0.5 * (pt1 + pt2)
    mid_pt /= np.linalg.norm(mid_pt)

    mid_pt_pix = TransformationSpherical.cartesian_to_pixel(mid_pt, width)

    dist_total = abs(pt1_pix[0, 0] - pt2_pix[0, 0])
    dist_left = abs(pt1_pix[0, 0] - mid_pt_pix[0, 0])
    dist_right = abs(pt2_pix[0, 0] - mid_pt_pix[0, 0])

    return dist_total > width / 2.0 or dist_left + dist_right > dist_total + 1

def split_index(width, vertices):
    num_vertices = vertices.shape[0]
    for i in range(1, num_vertices):
        pt1 = vertices[i-1]
        pt2 = vertices[i]
        
        if is_loop_closure_line(width, pt1, pt2):
            return i
    return -1


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
                            
                            idx = split_index(1024, xyz_floor)
                            if idx >= 0:
                                xyz_floor = np.concatenate([xyz_floor[idx:], xyz_floor[:idx]])
                                xyz_ceil = np.concatenate([xyz_ceil[idx:], xyz_ceil[:idx]])
                            
                            xyz_floor = xyz_floor[::-1]
                            xyz_ceil = xyz_ceil[::-1]                        
                            
                            xyzs = np.empty((xyz_floor.shape[0]*2, xyz_floor.shape[1]), dtype=xyz_floor.dtype)
                            xyzs[0::2] = xyz_ceil
                            xyzs[1::2] = xyz_floor

                            cor = TransformationSpherical.cartesian_to_pixel(xyzs, 1024).astype(int)                           

                            file_prefix = f'{floor_id}_{partial_room_id}_{pano_id}'
                            if pano_data['is_primary']:
                                file_prefix = f'{file_prefix}_primary'

                            txt_filename = os.path.join(label_dir, f'{file_prefix}.txt')
                            np.savetxt(txt_filename, cor, delimiter=' ', fmt='%d')
                            json_filename = os.path.join(label_dir, f'{file_prefix}.json')
                            with open(json_filename, 'w') as f: json.dump(pano_data, f)
                            num += 1
                            
    print(f'Number of Data: {num}')
    
