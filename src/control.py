"""Path tracking: a PI speed controller with feedforward and a Stanley
steering law.

The controller is given a short path [[x, y, speed], ...] and the car state,
and returns (throttle, steer, brake) in CARLA's ranges. It projects the front
axle onto the path continuously, so a coarse path does not make the steering
jump from point to point.
"""
import math

import numpy as np

from route import Route, wrap_angle


class PathController(object):
    def __init__(self, max_steer_deg=70.0, axle_offset=1.3, wheelbase=2.9, k_stanley=1.5,
                 soft_speed=1.0, kp=1.2, ki=0.4, steer_rate=3.0):
        self.max_steer = math.radians(max_steer_deg)
        self.axle_offset = axle_offset      # m from the reference point to the front axle
        self.wheelbase = wheelbase
        self.k_stanley = k_stanley
        self.soft_speed = soft_speed
        self.kp, self.ki = kp, ki
        self.steer_rate = steer_rate        # max change of the steer command per second
        self.path = None
        self.desired_speed = 0.0
        self._integral = 0.0
        self._steer = 0.0

    def set_path(self, waypoints):
        if len(waypoints) >= 2:
            self.path = Route(waypoints)

    def step(self, x, y, yaw, speed, dt):
        if self.path is None:
            return 0.0, 0.0, 1.0
        path = self.path

        # ---- steering: heading error plus cross track term at the front axle ----
        fx = x + self.axle_offset * math.cos(yaw)
        fy = y + self.axle_offset * math.sin(yaw)
        s, d, path_heading = path.project(fx, fy)
        heading_err = wrap_angle(path_heading - yaw)
        # curvature feedforward: the steering angle a bicycle needs to hold the path's bend
        kappa = wrap_angle(path.heading_at(s + 3.0) - path.heading_at(s - 1.0)) / 4.0   # 4 m window smooths the vertices
        delta = (heading_err + math.atan(self.wheelbase * kappa)
                 - math.atan2(self.k_stanley * d, self.soft_speed + max(speed, 0.0)))
        steer = float(np.clip(delta / self.max_steer, -1.0, 1.0))
        max_step = self.steer_rate * dt
        steer = float(np.clip(steer, self._steer - max_step, self._steer + max_step))
        self._steer = steer

        # ---- speed: look a little ahead of the reference point on the profile ----
        s_ref = path.project(x, y)[0]
        ahead = 0.5 + 0.3 * max(speed, 0.0)
        v_ref = path.limit_at(s_ref + ahead)
        slope = (path.limit_at(s_ref + ahead + 1.0) - path.limit_at(s_ref + ahead)) / 1.0
        a_ff = slope * max(speed, 0.5)          # dv/dt = v * dv/ds
        self.desired_speed = v_ref

        err = v_ref - speed
        if v_ref < 0.05:
            self._integral = 0.0
        else:
            self._integral = float(np.clip(self._integral + err * dt, -2.0, 2.0))
        a_cmd = self.kp * err + self.ki * self._integral + a_ff

        if v_ref < 0.05 and speed < 0.5:
            return 0.0, steer, 1.0              # hold still
        if a_cmd >= 0.0:
            return float(np.clip(a_cmd / 3.0, 0.0, 1.0)), steer, 0.0
        if a_cmd > -0.15:
            return 0.0, steer, 0.0              # coast, avoids chattering between pedals
        return 0.0, steer, float(np.clip(-a_cmd / 8.0, 0.0, 1.0))
