#!/usr/bin/env python3
"""Sensors on the ego car and a synchronised way to read them.

Pattern taken from the CARLA examples sensor_synchronization.py and
visualize_multiple_sensors.py: every sensor callback only drops its data into a
thread safe queue, and the main loop, right after world.tick(), collects the
data that belongs to that exact frame.

Every sensor is optional: if one cannot be created (blueprint or attribute
missing in your build) the rig prints a warning and carries on without it.
"""
import queue

import numpy as np

import carla
import perception as pc


class SensorRig(object):
    def __init__(self, world, bp_lib, ego, chase_size=None, front_size=None,
                 lidar=True, radar=True, lane_invasion=True, dt=0.05):
        self.world = world
        self.ego = ego
        self.sensors = []          # everything we spawned (for cleanup)
        self.lane_crossings = 0
        self.missed = 0
        self._queues = {}
        self._last = {}
        self._warned = set()
        ext = ego.bounding_box.extent
        self.ego_half_len, self.ego_half_wid = ext.x, ext.y
        self.radar_pos = np.array([ext.x + 0.05, 0.0, 0.6])
        self.front_fov = 90.0

        if chase_size is not None:
            self._add_camera(bp_lib, "chase", chase_size, 90.0,
                             carla.Transform(carla.Location(x=-8.5, z=3.2),
                                             carla.Rotation(pitch=-12.0)))
        if front_size is not None:
            p = pc.FRONT_CAM_POS
            self._add_camera(bp_lib, "front", front_size, self.front_fov,
                             carla.Transform(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))))
        if lidar:
            self._add_lidar(bp_lib, dt)
        if radar:
            self._add_radar(bp_lib)
        if lane_invasion:
            self._add_lane_invasion(bp_lib)

    # ------------------------------------------------------------------
    def _spawn(self, name, bp, transform):
        try:
            s = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        except Exception as exc:
            print("WARNING: could not create the %s sensor (%s), continuing without it" % (name, exc))
            return None
        self.sensors.append(s)
        return s

    def _listen(self, name, sensor):
        q = queue.Queue()
        self._queues[name] = q
        sensor.listen(q.put)

    def _add_camera(self, bp_lib, name, size, fov, transform):
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(int(size[0])))
        bp.set_attribute("image_size_y", str(int(size[1])))
        bp.set_attribute("fov", str(fov))
        s = self._spawn(name, bp, transform)
        if s is not None:
            self._listen(name, s)

    def _add_lidar(self, bp_lib, dt):
        bp = bp_lib.find("sensor.lidar.ray_cast")
        attrs = {"channels": "32", "range": "50", "points_per_second": "300000",
                 "rotation_frequency": str(1.0 / dt), "upper_fov": "10",
                 "lower_fov": "-30", "dropoff_general_rate": "0.0",
                 "dropoff_intensity_limit": "1.0", "dropoff_zero_intensity": "0.0"}
        for k, v in attrs.items():
            try:
                bp.set_attribute(k, v)
            except Exception:
                pass
        p = pc.LIDAR_POS
        s = self._spawn("lidar", bp, carla.Transform(
            carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))))
        if s is not None:
            self._listen("lidar", s)

    def _add_radar(self, bp_lib):
        bp = bp_lib.find("sensor.other.radar")
        for k, v in {"horizontal_fov": "30", "vertical_fov": "10", "range": "60",
                     "points_per_second": "4000"}.items():
            try:
                bp.set_attribute(k, v)
            except Exception:
                pass
        p = self.radar_pos
        s = self._spawn("radar", bp, carla.Transform(
            carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))))
        if s is not None:
            self._listen("radar", s)

    def _add_lane_invasion(self, bp_lib):
        try:
            bp = bp_lib.find("sensor.other.lane_invasion")
        except Exception as exc:
            print("WARNING: no lane invasion sensor (%s)" % exc)
            return
        s = self._spawn("lane invasion", bp, carla.Transform())
        if s is not None:
            s.listen(self._on_lane)

    def _on_lane(self, event):
        self.lane_crossings += 1

    # ------------------------------------------------------------------
    @property
    def has_lidar(self):
        return "lidar" in self._queues

    @property
    def has_radar(self):
        return "radar" in self._queues

    def collect(self, frame, timeout=20.0):
        """Return {name: raw sensor data} for this frame. Data that does not
        arrive in time is replaced by the previous frame's and counted."""
        out = {}
        for name, q in self._queues.items():
            data = None
            while True:
                try:
                    d = q.get(timeout=timeout)
                except queue.Empty:
                    break
                if d.frame >= frame:
                    data = d
                    break
            if data is None:
                self.missed += 1
                if name not in self._warned:
                    print("WARNING: no %s data for frame %s, reusing the last one" % (name, frame))
                    self._warned.add(name)
                data = self._last.get(name)
            else:
                self._last[name] = data
            out[name] = data
        return out

    def decode(self, data):
        """Raw sensor data -> numpy arrays in the ego frame (None if missing)."""
        res = {"chase": None, "front": None, "lidar": None, "radar": None}
        for cam in ("chase", "front"):
            d = data.get(cam)
            if d is not None:
                res[cam] = pc.decode_image(d.raw_data, d.height, d.width)
        d = data.get("lidar")
        if d is not None:
            res["lidar"] = pc.lidar_to_ego(pc.decode_lidar(d.raw_data))
        d = data.get("radar")
        if d is not None:
            res["radar"] = pc.radar_to_ego(
                ((x.velocity, x.azimuth, x.altitude, x.depth) for x in d), self.radar_pos)
        return res

    def stop_all(self):
        for s in self.sensors:
            try:
                s.stop()
            except Exception:
                pass
