import glob
import math
import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        "C:/CARLA_0.9.10/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.10",
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
from datetime import date
from modules import generator_KITTI as gen

CONFIG = {
    'lidar_extrinsics': {'x': 0.0, 'y': 0.0, 'z': 1.80, 'pitch': 0.0, 'yaw': 180.0, 'roll': 0.0},
    'camera_extrinsics': [
        {'x': 0.30, 'y': 0.0,   'z': 1.70, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
        {'x': 0.30, 'y': 0.50,  'z': 1.70, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
        {'x': 0.30, 'y': -0.50, 'z': 1.70, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
    ],
    'sequence_start': 60,
    'rgb_image_size_x': 1242,
    'rgb_image_size_y': 376,
    'rgb_fov': 90, # field of view on width
}

def main():
    start_record_full = time.time()

    fps_simu = 1000.0
    time_stop = 2.0
    nbr_frame = 5000 #MAX = 10000
    nbr_walkers = 50
    nbr_vehicles = 50

    actor_list = []
    vehicles_list = []
    all_walkers_id = []
    data_date = date.today().strftime("%Y_%m_%d")

    spawn_points = [23,46,0,125,53,257,62]

    init_settings = None

    try:
        client = carla.Client('localhost', 2000)
        init_settings = carla.WorldSettings()

        for i_map in [0, 1, 2, 3, 4, 5, 6]: #7 maps from Town01 to Town07
            client.set_timeout(100.0)
            print("Map Town0"+str(i_map+1))
            world = client.load_world("Town0"+str(i_map+1))
            folder_output = "KITTI_Dataset_CARLA_v%s/sequences/%02d" %(client.get_client_version(), CONFIG['sequence_start']+i_map)
            os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]
            client.start_recorder(os.path.dirname(os.path.realpath(__file__))+"/"+folder_output+"/recording.log")

            # Weather
            world.set_weather(carla.WeatherParameters.WetCloudyNoon)

            # Set Synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0/fps_simu
            settings.no_rendering_mode = False
            world.apply_settings(settings)

            # Create KITTI vehicle
            blueprint_library = world.get_blueprint_library()
            bp_KITTI = blueprint_library.find('vehicle.tesla.model3')
            bp_KITTI.set_attribute('color', '228, 239, 241')
            bp_KITTI.set_attribute('role_name', 'KITTI')
            start_pose = world.get_map().get_spawn_points()[spawn_points[i_map]]
            KITTI = world.spawn_actor(bp_KITTI, start_pose)
            waypoint = world.get_map().get_waypoint(start_pose.location)
            actor_list.append(KITTI)
            print('Created %s' % KITTI)

            # Spawn vehicles and walkers
            gen.spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id)

            # Wait for KITTI to stop
            start = world.get_snapshot().timestamp.elapsed_seconds
            print("Waiting for KITTI to stop ...")
            while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop: world.tick()
            print("KITTI stopped")

            # Set sensors transformation from KITTI
            lidar_transform = carla.Transform(
                carla.Location(
                    x=CONFIG['lidar_extrinsics']['x'],
                    y=CONFIG['lidar_extrinsics']['y'],
                    z=CONFIG['lidar_extrinsics']['z']
                ),
                carla.Rotation(
                    pitch=CONFIG['lidar_extrinsics']['pitch'],
                    yaw=CONFIG['lidar_extrinsics']['yaw'],
                    roll=CONFIG['lidar_extrinsics']['roll']
                )
            )
            cam_transforms = [
                carla.Transform(
                    carla.Location(
                        x=CONFIG['camera_extrinsics'][i]['x'],
                        y=CONFIG['camera_extrinsics'][i]['y'],
                        z=CONFIG['camera_extrinsics'][i]['z']
                    ),
                    carla.Rotation(
                        pitch=CONFIG['camera_extrinsics'][i]['pitch'],
                        yaw=CONFIG['camera_extrinsics'][i]['yaw'],
                        roll=CONFIG['camera_extrinsics'][i]['roll']
                    )
                ) for i in range(len(CONFIG['camera_extrinsics']))
            ]

            # Take a screenshot
            gen.screenshot(KITTI, world, actor_list, folder_output, carla.Transform(carla.Location(x=0.0, y=0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0)), CONFIG)

            # Create our sensors
            gen.RGB.sensor_id_glob = 0
            gen.HDL64E.sensor_id_glob = 100
            VelodyneHDL64 = gen.HDL64E(KITTI, world, actor_list, folder_output, lidar_transform, CONFIG)
            cams = [gen.RGB(KITTI, world, actor_list, folder_output, cam_transforms[i], CONFIG) for i in range(len(cam_transforms))]

            # Export LiDAR to camera transformations
            Ks = []
            Trs = []
            to_string = lambda array: ' '.join(map(str, array.flatten().tolist()))
            for i, cam_transform in enumerate(cam_transforms):
                focal_length = CONFIG['rgb_image_size_x'] / (2 * math.tan(CONFIG['rgb_fov'] * math.pi / 360))
                center_x = CONFIG['rgb_image_size_x'] / 2
                center_y = CONFIG['rgb_image_size_y'] / 2
                Ks.append(np.array([[focal_length, 0, center_x],
                                    [0, focal_length, center_y],
                                    [0, 0, 1]]))

                R = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]).reshape(3, 3)
                R = np.hstack([R, np.zeros((3, 1))])
                R = np.vstack([R, np.array([0.0, 0.0, 0.0, 1.0])])

                Tr = gen.transform_lidar_to_camera(lidar_transform, cam_transform)
                Tr = np.dot(R, Tr)
                Tr = Tr[:3, :4]
                Trs.append(Tr)

            # KITTI style calibration file (only supports one camera)
            with open(folder_output+f"/calib.txt", 'w') as f:
                P = np.hstack([Ks[0], np.zeros((3, 1))])
                for i in range(4):
                    f.write(f'P{i}: {to_string(P)}\n')
                f.write(f'Tr: {to_string(Trs[0])}')

            # New style calibration file
            with open(folder_output+f"/calib2.txt", 'w') as posfile:
                for i, K in enumerate(Ks):
                    posfile.write(f'K{i}: {to_string(K)}\n')
                for i, Tr in enumerate(Trs):
                    posfile.write(f'Tr{i}: {to_string(Tr)}\n')


            # Launch KITTI
            KITTI.set_autopilot(True)

            # Pass to the next simulator frame to spawn sensors and to retrieve first data
            world.tick()

            VelodyneHDL64.init()
            gen.follow(KITTI.get_transform(), world)

            # All sensors produce first data at the same time (this ts)
            gen.Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds

            start_record = time.time()
            print("Start record : ")
            frame_current = 0
            while (frame_current < nbr_frame):
                frame_current = VelodyneHDL64.save()
                for cam in cams:
                    cam.save()
                gen.follow(KITTI.get_transform(), world)
                world.tick()    # Pass to the next simulator frame

            VelodyneHDL64.save_poses()
            client.stop_recorder()
            print("Stop record")

            print('Destroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list.clear()

            # Stop walker controllers (list is [controller, actor, controller, actor ...])
            all_actors = world.get_actors(all_walkers_id)
            for i in range(0, len(all_walkers_id), 2):
                all_actors[i].stop()
            print('Destroying %d walkers' % (len(all_walkers_id)//2))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_walkers_id])
            all_walkers_id.clear()

            print('Destroying KITTI')
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            actor_list.clear()

            print("Elapsed time : ", time.time()-start_record)
            print()

            time.sleep(2.0)

    finally:
        print("Elapsed total time : ", time.time()-start_record_full)
        world.apply_settings(init_settings)

        time.sleep(2.0)


if __name__ == '__main__':
    main()
