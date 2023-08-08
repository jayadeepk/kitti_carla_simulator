#!/usr/bin/env python3
"""
Convert KITTI-CARLA dataset into KITTI odometry dataset format
"""

from modules import ply
from PIL import Image
from tqdm import tqdm
import math
import numpy as np
import os
import shutil

KITTI_CARLA_DIR = '/home/jay/datasets/KITTI-CARLA'
OUTPUT_DIR = '/home/jay/datasets/KITTI-CARLA-KITTI-format'

CAMERA_FOV = 72
IMAGE_SIZE_X = 1392
IMAGE_SIZE_Y = 1024
TARGET_IMAGE_SIZE_X = 1242
TARGET_IMAGE_SIZE_Y = 376


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

def compute_intrinsics():
    focal_length = IMAGE_SIZE_X / (2 * math.tan(CAMERA_FOV * math.pi / 360))
    center_x = TARGET_IMAGE_SIZE_X / 2
    center_y = TARGET_IMAGE_SIZE_Y / 2
    return np.array([[focal_length, 0, center_x],
                     [0, focal_length, center_y],
                     [0, 0, 1]])


def crop_image(image_path, config):
                    """
                    Crop the image to the desired size
                    """
                    img = Image.open(image_path)
                    width, height = img.size
                    left = (width - config['target_image_size_x']) / 2
                    top = (height - config['target_image_size_y']) / 2
                    right = (width + config['target_image_size_x']) / 2
                    bottom = (height + config['target_image_size_y']) / 2
                    img = img.crop((left, top, right, bottom))
                    return img


def convert_to_kitti_odometry_format(kitti_carla_dir, output_dir, config):
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
                cropped_img = crop_image(image_path, config)
                cropped_img.save(os.path.join(output_image_left_dir, image.replace('_0.png', '.png')))
            elif image.endswith('_1.png'):
                cropped_img = crop_image(image_path, config)
                cropped_img.save(os.path.join(output_image_right_dir, image.replace('_1.png', '.png')))

        # Convert timestamps file
        print('  Timestamps')
        with open(os.path.join(input_town_dir, 'generated', 'full_ts_camera.txt'), 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            lines = [line.split(' ')[1] for line in lines]

        with open(os.path.join(output_sequences_dir, town, 'times.txt'), 'w') as f:
            f.writelines(lines)

        # Generate calibration file
        print('  Calibration')
        K = compute_intrinsics()
        Ps = [np.hstack([K, np.zeros((3, 1))])] * 4

        with open(os.path.join(input_town_dir, 'generated', 'lidar_to_cam0.txt'), 'r') as f:
            Tr = f.readlines()[1]

            R = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]).reshape(3, 3)
            R = np.hstack([R, np.zeros((3, 1))])
            R = np.vstack([R, np.array([0.0, 0.0, 0.0, 1.0])])

            Tr = Tr.strip().split(' ')
            Tr = np.array([float(x) for x in Tr]).reshape(3, 4)
            Tr = np.vstack([Tr, np.array([0.0, 0.0, 0.0, 1.0])])

            # Tr = R x Tr
            Tr = np.dot(R, Tr)
            Tr = Tr[:3, :4]
            Tr = Tr.flatten().tolist()
            Tr = ' '.join(map(str, Tr))

        with open(os.path.join(output_sequences_dir, town, 'calib.txt'), 'w') as f:
            for i, P in enumerate(Ps):
                f.write(f'P{i}: {" ".join(map(str, P.reshape(-1)))}\n')
            # TODO: Tr_velo_to_cam is currently set to left color camera due
            #       to differences in KITTI and KITTI-CARLA coordinate
            #       systems. So this calib file would not work for other
            #       cameras.
            f.write('Tr: ' + Tr)


if __name__ == '__main__':
    config = {
        'target_image_size_x': TARGET_IMAGE_SIZE_X,
        'target_image_size_y': TARGET_IMAGE_SIZE_Y,
    }
    convert_to_kitti_odometry_format(KITTI_CARLA_DIR, OUTPUT_DIR, config)
