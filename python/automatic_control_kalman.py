# -*- coding: utf-8 -*-

"""Revised automatic control
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT
import numpy as np
import os
import math
import random
import queue
import sys
import matplotlib.pyplot as plt
import time
import carla
from python.agents.navigation.behavior_agent import BehaviorAgent



gnss_data = {'lat': None, 'lon': None, 'alt': None, 'trans': None}
def gnss_callback(gnss):
    gnss_data['lat'] = gnss.latitude
    gnss_data['lon'] = gnss.longitude
    gnss_data['alt'] = gnss.altitude
    gnss_data['trans'] = gnss.transform


latest_imu_data = {"accel": None, "gyro": None, "compass": None}
def imu_callback(imu_data):
    latest_imu_data["accel"] = imu_data.accelerometer
    latest_imu_data["gyro"] = imu_data.gyroscope
    latest_imu_data["compass"] = imu_data.compass

def skew_symmetric(v):
    """Skew symmetric form of a 3x1 vector."""
    return np.array(
        [[0, -v[2][0], v[1][0]],
         [v[2][0], 0, -v[0][0]],
         [-v[1][0], v[0][0], 0]], dtype=np.float64)


class Quaternion():
    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        """
        Allow initialization with explicit quaterion wxyz, axis-angle, or Euler XYZ (RPY) angles.

        :param w: w (real) of quaternion.
        :param x: x (i) of quaternion.
        :param y: y (j) of quaternion.
        :param z: z (k) of quaternion.
        :param axis_angle: Set of three values from axis-angle representation, as list or [3,] or [3,1] np.ndarray.
                           See C2M5L2 for details.
        :param euler: Set of three XYZ Euler angles. 
        """
        if axis_angle is None and euler is None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        elif euler is not None and axis_angle is not None:
            raise AttributeError("Only one of axis_angle or euler can be specified.")
        elif axis_angle is not None:
            if not (type(axis_angle) == list or type(axis_angle) == np.ndarray) or len(axis_angle) != 3:
                raise ValueError("axis_angle must be list or np.ndarray with length 3.")
            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.w = np.cos(norm / 2)
            if norm < 1e-50:  # to avoid instabilities and nans
                self.x = 0
                self.y = 0
                self.z = 0
            else:
                imag = axis_angle / norm * np.sin(norm / 2)
                self.x = imag[0].item()
                self.y = imag[1].item()
                self.z = imag[2].item()
        else:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)

            # Fixed frame
            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

            # Rotating frame
            # self.w = cr * cp * cy - sr * sp * sy
            # self.x = cr * sp * sy + sr * cp * cy
            # self.y = cr * sp * cy - sr * cp * sy
            # self.z = cr * cp * sy + sr * sp * cy

    def __repr__(self):
        return "Quaternion (wxyz): [%2.5f, %2.5f, %2.5f, %2.5f]" % (self.w, self.x, self.y, self.z)

    def to_axis_angle(self):
        t = 2*np.arccos(self.w)
        return np.array(t*np.array([self.x, self.y, self.z])/np.sin(t/2))

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w ** 2 - np.dot(v.T, v)) * np.eye(3) + \
               2 * np.dot(v, v.T) + 2 * self.w * skew_symmetric(v)

    def to_euler(self):
        """Return as xyz (roll pitch yaw) Euler angles."""
        roll = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))
        pitch = np.arcsin(2 * (self.w * self.y - self.z * self.x))
        yaw = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))
        return np.array([roll, pitch, yaw])

    def to_numpy(self):
        """Return numpy wxyz representation."""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        """Return a (unit) normalized version of this quaternion."""
        norm = np.linalg.norm([self.w, self.x, self.y, self.z])
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def quat_mult_right(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the right, that is, q*self.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = -skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

    def quat_mult_left(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the left, that is, self*q.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj
var_imu_f = 0.05
var_imu_w = 0.001
var_gnss  = 0.5
var_lidar = 0.2
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3) 
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    r_cov = np.eye(3) * sensor_var
    k_gain = p_cov_check.dot(h_jac.T).dot(np.linalg.inv(h_jac.dot(p_cov_check).dot(h_jac.T) + r_cov))
    # 3.2 Compute error state
    x_error = k_gain.dot(y_k - p_check)
    # 3.3 Correct predicted state
    p_hat = p_check + x_error[0:3]
    v_hat = v_check + x_error[3:6]
    q_hat = Quaternion(euler=x_error[6:9]).quat_mult_left(q_check)
    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9) - k_gain.dot(h_jac)).dot(p_cov_check)
    return p_hat, v_hat, q_hat, p_cov_hat

def init_kalman(vehicle):
    velocity_vector = vehicle.get_velocity()
    rotation = vehicle.get_transform().rotation
    rot = np.array([rotation.roll, rotation.pitch, rotation.yaw])
    transform = vehicle.get_transform()
    location = transform.location
    p_est = np.array([location.x, location.y, location.z])
    v_est = np.array([velocity_vector.x,velocity_vector.y,velocity_vector.z])
    q_est = Quaternion(euler=rot).to_numpy()
    p_cov = np.zeros(9)
    return p_est, v_est, q_est, p_cov
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
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # read all valid spawn points
        all_default_spawn = world.get_map().get_spawn_points()
        # randomly choose one as the start point
        # spawn_point = random.choice(all_default_spawn) if all_default_spawn else carla.Transform()
        spawn_point = all_default_spawn[0]
        # create the blueprint library
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)
        imu_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
        # attach the IMU sensor to the vehicle
        imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        imu_sensor.listen(lambda data: imu_callback(data))

        # attach gnss sensor to the vehicle
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.05')  # 每 0.05 秒更新一次
        gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2.0))  # 安装在车顶附近
        gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss_sensor.listen(gnss_callback)
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
        his_trajectory = []
        p_est_trajectory = []
        i = 0
        start_flag = True
        while True:
        # for i in range(200):
            i += 1
            delta_t = settings.fixed_delta_seconds
            agent.update_information(vehicle)

            # the imu data is not stable at the beginning (e.g. -32556,235235,5435), so we skip the first 30 frames
            if i > 30:
                if start_flag:
                    p_est, v_est, q_est, p_cov  = init_kalman(vehicle)
                    start_flag = False
                his_trajectory.append(vehicle.get_transform().location)
                p_est_trajectory.append(p_est)

            imu_f = np.array([latest_imu_data['accel'].x,latest_imu_data['accel'].y,latest_imu_data['accel'].z])
            imu_w = np.array([latest_imu_data['gyro'].x,latest_imu_data['gyro'].y,latest_imu_data['gyro'].z])

            gnss_p = np.array([gnss_data['trans'].location.x,gnss_data['trans'].location.y,gnss_data['trans'].location.z])
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
            
            if i > 30:
                C_ns = Quaternion(*q_est).to_mat()
                C_ns_d_f_km = np.dot(C_ns, imu_f) # from ego to world frame
                p_est = p_est + delta_t * v_est + (delta_t**2) / 2 * (C_ns_d_f_km + g)
                v_est = v_est + delta_t * (C_ns_d_f_km + g)
                # get current rotation
                q_fr_w = Quaternion(axis_angle=imu_w * delta_t)
                q_est = q_fr_w.quat_mult_right(q_est)

                # 1.1 Linearize the motion model and compute Jacobians
                f_ja_km = np.identity(9)
                f_ja_km[0:3, 3:6] = np.identity(3) * delta_t
                f_ja_km[3:6, 6:9] = -skew_symmetric(np.array(C_ns_d_f_km).reshape(len(C_ns_d_f_km),1)) * delta_t

                # 2. Propagate uncertainty
                q_cov_km = np.identity(6)
                q_cov_km[0:3,0:3] *=  delta_t**2 * np.eye(3) * var_imu_f
                q_cov_km[3:6, 3:6] *= delta_t**2 * np.eye(3) * var_imu_w
                p_cov = f_ja_km.dot(p_cov).dot(f_ja_km.T) + l_jac.dot(q_cov_km).dot(l_jac.T)

                # if gnss_i < gnss.data.shape[0] and imu_f.t[k] == gnss.t[gnss_i]:
                p_est, v_est, q_est, p_cov = \
                    measurement_update(var_gnss, p_cov, gnss_p, p_est, v_est, q_est)
                    

            # time.sleep(0.1)
        x_vals = [loc.x for loc in his_trajectory]
        y_vals = [loc.y for loc in his_trajectory]

        est_x_vals = [loc[0] for loc in p_est_trajectory]
        est_y_vals = [loc[1] for loc in p_est_trajectory]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, linestyle='-', color='blue', label='Ground Truth')
        plt.plot(est_x_vals, est_y_vals, linestyle='--', color='red', label='Estimated')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Vehicle Trajectory: Ground Truth vs Estimated')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # 保持 x 和 y 比例一致
        plt.savefig("trajectory_comparison.png", dpi=300)
        plt.show()
    except Exception as e:
        print('something wrong', e)
    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()
        imu_sensor.destroy()
        gnss_sensor.destroy()
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
