# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla
from agents.tools.misc import get_speed


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)
        # self._lat_controller = PureSuitLateralController(self._vehicle)
        # self._lat_controller = StanleyLateralController(self._vehicle)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """
        # Add feedforward controller, the data can be obtained by simulation.
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        if target_speed <=6:
            ff = 0.15 + target_speed/6*(0.6-0.15)
        elif target_speed <=11.5:
            ff = 0.6 + (target_speed-6)/(11.5-6)*(0.8-0.6)
        else:
            ff = 0.8+(target_speed-11.5)/85
        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie) + ff, -1.0, 1.0)


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x -
                          v_begin.x, waypoint.transform.location.y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class PureSuitLateralController():
    def __init__(self, vehicle, offset=0):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        physics_control = self._vehicle.get_physics_control()
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
        self._wheelbase = np.linalg.norm(front_pos - rear_pos)/100

        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pure_suit_control(waypoint, self._vehicle.get_transform())

    def set_offset(self, offset):
        """Changes the offset"""
        self._offset = offset

    def _pure_suit_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = waypoint.transform.location

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])
        physics_control = self._vehicle.get_physics_control()
        wheels = physics_control.wheels

        # 获取后轴中心点（两后轮平均）
        rear_pos = np.array([
            (wheels[2].position.x + wheels[3].position.x) / 2.0,
            (wheels[2].position.y + wheels[3].position.y) / 2.0,
            0,
        ])
        Ld = np.linalg.norm(np.array([w_loc.x, w_loc.y , 0.0]) - rear_pos/100)
        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            alpha = 1
        else:
            alpha = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            alpha *= -1.0

        steer_rad = math.atan2(2.0 * self._wheelbase * math.sin(alpha), Ld)


        # return np.clip(steer_rad, -1.0, 1.0)
        return steer_rad
    
class StanleyLateralController():
    def __init__(self, vehicle, offset=0):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        physics_control = self._vehicle.get_physics_control()
        wheels = physics_control.wheels
        self._past_wp = None
        self._tmp_wp = None
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
        self._wheelbase = np.linalg.norm(front_pos - rear_pos)/100

        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._stanley_control(waypoint, self._vehicle.get_transform())

    def set_offset(self, offset):
        """Changes the offset"""
        self._offset = offset

    def _stanley_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        if self._past_wp is None:
            self._past_wp = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = waypoint.transform.location

        path_vec = np.array([w_loc.x - self._past_wp.x ,
                          w_loc.y - self._past_wp.y,
                          0.0])
        path_yaw = math.atan2(path_vec[1], path_vec[0])
        
        if self._tmp_wp is not None and self._tmp_wp.x != w_loc.x and self._tmp_wp.y != w_loc.y:
            self._past_wp = self._tmp_wp
        # Ld = np.linalg.norm(np.array([w_loc.x, w_loc.y , 0.0]) - front_pos/100)
        wv_linalg = np.linalg.norm(path_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            heading_error = 1
        else:
            heading_error = math.acos(np.clip(np.dot(path_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, path_vec)
        if _cross[2] < 0:
            heading_error *= -1.0
 
        ego_loc = vehicle_transform.location
        yaw = math.radians(vehicle_transform.rotation.yaw)
        front_point = np.array([
            ego_loc.x + self._wheelbase / 2 * math.cos(yaw),
            ego_loc.y + self._wheelbase / 2 * math.sin(yaw),
            0.0
        ])
        closest_point = np.array([w_loc.x,
                          w_loc.y,
                          0.0])
        error_vec = front_point - closest_point
        normal_vec = np.array([-math.sin(path_yaw), math.cos(path_yaw),0])
        cross_track_error = np.dot(error_vec, normal_vec)
        self._tmp_wp = waypoint.transform.location
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        delta = heading_error + math.atan2(2.5 * cross_track_error, speed + 1e-3)
        return np.clip(delta, -1.0, 1.0)
