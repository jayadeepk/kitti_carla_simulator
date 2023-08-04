#!/usr/bin/env python3

from modules import ply
from tqdm import tqdm
import numpy as np
import os
import shutil

KITTI_CARLA_DIR = '/home/jay/datasets/KITTI-CARLA'
OUTPUT_DIR = '/home/jay/datasets/KITTI-CARLA-KITTI-format'


def convert_ply_to_bin(ply_dir, output_bin_dir):
    """
    Convert .ply LiDAR point cloud files to a KITTI .bin files
    """
    ply_field_names = ['x', 'y', 'z', 'cos_angle_lidar_surface']
    os.makedirs(os.path.dirname(output_bin_dir), exist_ok=True)
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
    for ply_file in tqdm(ply_files, desc='    '):
        ply_file = os.path.join(ply_dir, ply_file)
        data = ply.read_ply(ply_file)
        xyzc = np.vstack([data[axis] for axis in ply_field_names[:4]]).T     # (N, 4)
        output_file = os.path.basename(ply_file).lstrip('frame_').rstrip('.ply') + '.bin'
        xyzc.astype('float32').tofile(os.path.join(output_bin_dir, output_file))


def convert_kitti_carla_to_kitti_format(kitti_carla_dir, output_dir):
    output_sequences_dir = os.path.join(output_dir, 'sequences')
    for town in os.listdir(kitti_carla_dir):
        print(town)
        input_town_dir = os.path.join(kitti_carla_dir, town)
        output_velodyne_dir = os.path.join(output_sequences_dir, town, 'velodyne')
        output_image_left_dir = os.path.join(output_sequences_dir, town, 'image_2')
        output_image_right_dir = os.path.join(output_sequences_dir, town, 'image_3')

        os.makedirs(output_velodyne_dir, exist_ok=True)
        os.makedirs(output_image_left_dir, exist_ok=True)
        os.makedirs(output_image_right_dir, exist_ok=True)

        ply_dir = os.path.join(input_town_dir, 'generated', 'frames')

        # Convert .ply LiDAR point cloud files to KITTI .bin files
        print('  LiDAR')
        convert_ply_to_bin(ply_dir, output_velodyne_dir)

        # Copy KITTI CARLA stereo images to the output directory
        print('  Images')
        images_dir = os.path.join(input_town_dir, 'generated', 'images_rgb')
        for image in tqdm(os.listdir(images_dir), desc='    '):
            image_path = os.path.join(images_dir, image)
            if image.endswith('_0.png'):
                shutil.copy(image_path, output_image_left_dir)
            if image.endswith('_1.png'):
                shutil.copy(image_path, output_image_right_dir)


if __name__ == '__main__':
    convert_kitti_carla_to_kitti_format(KITTI_CARLA_DIR, OUTPUT_DIR)
