import glob
import math
import numpy as np
import os
import random
import sys
import time
import yaml

from modules import sensors
from modules import objects
from modules.common import add_carla_to_path; add_carla_to_path()
import carla


DEFAULT_CONFIG_FILE = 'config/default.yaml'

def main(config):
    start_record_full = time.time()

    fps_simu = 1000.0
    time_stop = 2.0
    nbr_walkers = 50
    nbr_vehicles = 50

    actor_list = []
    vehicles_list = []
    all_walkers_id = []
    init_settings = None

    try:
        client = carla.Client('localhost', 2000)
        init_settings = carla.WorldSettings()

        for i, i_town in enumerate(config['towns']):
            client.set_timeout(100.0)
            print("Map Town0"+str(i_town))
            world = client.load_world("Town0"+str(i_town))
            folder_output = "SiMCaL_%s/sequences/%02d" % (config['version'], config['sequence_start'] + i)
            os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]
            client.start_recorder(os.path.dirname(os.path.realpath(__file__))+"/"+folder_output+"/recording.log")

            # Weather
            world.set_weather(carla.WeatherParameters.ClearNoon)

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
            start_pose = world.get_map().get_spawn_points()[config['spawn_points'][i]]
            KITTI = world.spawn_actor(bp_KITTI, start_pose)
            waypoint = world.get_map().get_waypoint(start_pose.location)
            actor_list.append(KITTI)
            print('Created %s' % KITTI)

            # Spawn vehicles and walkers
            objects.spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id)

            # Wait for KITTI to stop
            start = world.get_snapshot().timestamp.elapsed_seconds
            print("Waiting for KITTI to stop ...")
            while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop: world.tick()
            print("KITTI stopped")

            # Set sensors transformation from KITTI
            lidar_transform = carla.Transform(
                carla.Location(
                    x=config['lidar_extrinsics']['x'],
                    y=config['lidar_extrinsics']['y'],
                    z=config['lidar_extrinsics']['z']
                ),
                carla.Rotation(
                    pitch=config['lidar_extrinsics']['pitch'],
                    yaw=config['lidar_extrinsics']['yaw'],
                    roll=config['lidar_extrinsics']['roll']
                )
            )
            cam_transforms = [
                carla.Transform(
                    carla.Location(
                        x=config['cameras'][i]['extrinsics']['x'],
                        y=config['cameras'][i]['extrinsics']['y'],
                        z=config['cameras'][i]['extrinsics']['z']
                    ),
                    carla.Rotation(
                        pitch=config['cameras'][i]['extrinsics']['pitch'],
                        yaw=config['cameras'][i]['extrinsics']['yaw'],
                        roll=config['cameras'][i]['extrinsics']['roll']
                    )
                ) for i in range(len(config['cameras']))
            ]

            # Create our sensors
            sensors.RGB.sensor_id_glob = 0
            sensors.HDL64E.sensor_id_glob = 100
            VelodyneHDL64 = sensors.HDL64E(KITTI, world, actor_list, folder_output, lidar_transform)
            cams = [sensors.RGB(KITTI, world, actor_list, folder_output, cam_transforms[i], config['cameras'][i]) for i in range(len(cam_transforms))]

            # Export LiDAR to camera transformations
            Ks = []
            Trs = []
            to_string = lambda array: ' '.join(map(str, array.flatten().tolist()))
            for i, cam_transform in enumerate(cam_transforms):
                focal_length = config['cameras'][i]['width'] / (2 * math.tan(config['cameras'][i]['fov'] * math.pi / 360))
                center_x = config['cameras'][i]['width'] / 2
                center_y = config['cameras'][i]['height'] / 2
                Ks.append(np.array([[focal_length, 0, center_x],
                                    [0, focal_length, center_y],
                                    [0, 0, 1]]))
                Trs.append(sensors.transform_lidar_to_camera(lidar_transform, cam_transform))

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
            sensors.follow(KITTI.get_transform(), world)

            # All sensors produce first data at the same time (this ts)
            sensors.Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds

            start_record = time.time()
            print("Start record : ")
            frame_current = 0
            random.seed(0)
            while (frame_current < config['frames_per_town']):
                if frame_current % config['weather_update_frequency'] == 0:
                    weather = carla.WeatherParameters(cloudiness=float(random.randint(5, 40)),
                                                      sun_altitude_angle=float(random.randint(15, 90)),
                                                      sun_azimuth_angle=float(random.randint(80, 100)),
                                                      fog_density=float(random.randint(0, 2)),
                                                      fog_distance=0.75,
                                                      fog_falloff=0.1,
                                                      scattering_intensity=1.0,
                                                      mie_scattering_scale=0.03,
                                                      rayleigh_scattering_scale=0.0331)
                    world.set_weather(weather)

                frame_current = VelodyneHDL64.save()
                for cam in cams:
                    cam.save()
                sensors.follow(KITTI.get_transform(), world)
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
    config_file = DEFAULT_CONFIG_FILE
    if len(sys.argv) > 1:
        if not os.path.isfile(sys.argv[1]):
            print(f'Error: Config file "{sys.argv[1]}" does not exist.')
            exit(1)
        config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
