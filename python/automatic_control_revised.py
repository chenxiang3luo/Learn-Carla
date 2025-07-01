# -*- coding: utf-8 -*-

"""Revised automatic control
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT
import numpy as np
import os
import math
import random
import sys
import time
import carla

from python.agents.navigation.behavior_agent import BehaviorAgent


def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Retrieve the world that is currently running
        world = client.get_world()

        origin_settings = world.get_settings()

        # set sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # read all valid spawn points
        all_default_spawn = world.get_map().get_spawn_points()
        # randomly choose one as the start point
        # spawn_point = random.choice(all_default_spawn) if all_default_spawn else carla.Transform()
        spawn_point = all_default_spawn[0]
        # create the blueprint library
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

        # calculate basic parameters
        physics_control = vehicle.get_physics_control()
        wheels = physics_control.wheels

        # 获取前轮和后轮的位置（以车辆坐标为参考）
        front_pos = np.array([
            (wheels[0].position.x + wheels[1].position.x) / 2.0,
            (wheels[0].position.y + wheels[1].position.y) / 2.0,
        ])

        # 获取后轴中心点（两后轮平均）
        rear_pos = np.array([
            (wheels[2].position.x + wheels[3].position.x) / 2.0,
            (wheels[2].position.y + wheels[3].position.y) / 2.0,
        ])

        # 欧几里得距离
        wheelbase = np.linalg.norm(front_pos - rear_pos)/100
 
        print(f"Estimated wheelbase: {wheelbase:.3f} meters")

        # 获取最大的 steering angle（单位：radians）
        # max_steer_rad = max(p.x for p in steering_curve)
        # print(f"Max steering angle: {math.degrees(max_steer_rad):.2f} degrees")
        # we need to tick the world once to let the client update the spawn position
        world.tick()

        # create the behavior agent
        agent = BehaviorAgent(vehicle, behavior='normal')

        # set the destination spot
        waypoint = world.get_map().get_waypoint(spawn_point.location)
        # random.shuffle(spawn_points)

        # to avoid the destination and start position same
        # if spawn_points[0].location != agent.vehicle.get_location():
        #     destination = spawn_points[0]
        # else:
        #     destination = spawn_points[1]
        target_waypoint = waypoint.next(500)[-1]

        destination = target_waypoint.transform

        # generate the route
        agent.set_destination(agent.vehicle.get_location(), destination.location, clean=True)

        while True:
            agent.update_information(vehicle)

            world.tick()
            
            if len(agent._local_planner.waypoints_queue)<1:
                print('======== Success, Arrivied at Target Point!')
                break
                
            # top view
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                    carla.Rotation(pitch=-90)))

            speed_limit = vehicle.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step(debug=True)
            vehicle.apply_control(control)
            time.sleep(0.1)
    except Exception as e:
        print('something wrong', e)
    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
