#!/usr/bin/env python3
"""Sensor processing for the motion planning demo. Pure NumPy, no carla import,
so everything here can be tested without a simulator.

Frames used in this file
------------------------
ego frame : origin at the car's reference point, x forward, y to the RIGHT,
            z up (the same handedness CARLA/Unreal use, so a top down plot with
            "forward = up" has "right = right").
world frame: CARLA world coordinates, yaw measured clockwise seen from above.
"""
import math

import numpy as np

# Mounting points, metres in the ego frame (x forward, y right, z up).
LIDAR_POS = np.array([0.0, 0.0, 2.0])
FRONT_CAM_POS = np.array([1.5, 0.0, 1.6])

# --- safety brake (AEB) tuning ------------------------------------------------
AEB_DECEL = 6.0          # m/s^2 assumed braking capability
AEB_MARGIN = 1.0         # m extra standing gap
AEB_REACTION = 0.15      # s
AEB_HALF_WIDTH = 1.25    # m, half width of the swept corridor around the path
AEB_Z_MIN, AEB_Z_MAX = 0.35, 2.2   # m above ground, ignores road surface and bridges
AEB_RELEASE = 1.25       # hysteresis factor on the release distance

# --- radar lead tracking --------------------------------------------------------
RADAR_CORRIDOR = 2.0     # m lateral distance to the planned path
RADAR_MIN_GROUND_SPEED = 0.8   # m/s, slower than this counts as a static object
LEAD_CENTRE_OFFSET = 2.4       # m from the lead's rear bumper to the point the planner expects
RADAR_HOLD_FRAMES = 6    # keep the last estimate this many frames after losing it


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def decode_image(raw, height, width):
    """CARLA BGRA buffer -> (H, W, 3) uint8 RGB."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return np.ascontiguousarray(arr[:, :, 2::-1])


def decode_lidar(raw):
    """Lidar buffer -> (N, 4) float32 [x, y, z, intensity] in the sensor frame."""
    pts = np.frombuffer(raw, dtype=np.float32)
    return pts.reshape((-1, 4)).copy()


def lidar_to_ego(pts):
    """Sensor frame -> ego frame (the lidar has no rotation on the car)."""
    out = pts.copy()
    out[:, 0:3] += LIDAR_POS.astype(np.float32)
    return out


def radar_to_ego(dets, radar_pos):
    """dets: iterable of (velocity, azimuth, altitude, depth) -> (M, 4) array
    [x, y, z, velocity] in the ego frame. velocity < 0 means closing in."""
    d = np.asarray(list(dets), dtype=np.float64).reshape((-1, 4))
    if d.size == 0:
        return np.zeros((0, 4))
    vel, az, alt, depth = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    x = depth * np.cos(alt) * np.cos(az) + radar_pos[0]
    y = depth * np.cos(alt) * np.sin(az) + radar_pos[1]
    z = depth * np.sin(alt) + radar_pos[2]
    return np.stack([x, y, z, vel], axis=1)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def world_to_ego(pts_xy, x, y, yaw):
    """(N, 2) world points -> (N, 2) ego frame [forward, right]."""
    pts = np.asarray(pts_xy, dtype=float).reshape((-1, 2))
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def ego_to_world(pts_xy, x, y, yaw):
    pts = np.asarray(pts_xy, dtype=float).reshape((-1, 2))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1],
                     y + s * pts[:, 0] + c * pts[:, 1]], axis=1)


def densify(path_xy, step=0.25):
    """Resample a polyline to roughly `step` metre spacing.
    Returns (points (K,2), arclength (K,))."""
    p = np.asarray(path_xy, dtype=float).reshape((-1, 2))
    if len(p) < 2:
        return p, np.zeros(len(p))
    seg = np.hypot(*(np.diff(p, axis=0).T))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return p[:1], np.zeros(1)
    ss = np.arange(0.0, s[-1], step)
    ss = np.append(ss, s[-1])
    return np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], axis=1), ss


def box_corners(cx, cy, yaw, half_len, half_wid, ox=0.0, oy=0.0):
    """Four corners (4, 2) of an oriented box, order: front-right, front-left,
    rear-left, rear-right (ego/world frames share this handedness)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for lx, ly in [(half_len, half_wid), (half_len, -half_wid),
                   (-half_len, -half_wid), (-half_len, half_wid)]:
        lx += ox
        ly += oy
        out.append([cx + c * lx - s * ly, cy + s * lx + c * ly])
    return np.array(out)


# ----------------------------------------------------------------------------
# Safety brake (AEB) on the lidar
# ----------------------------------------------------------------------------
def path_clearance(lidar_ego, path_ego, front_offset, half_width=AEB_HALF_WIDTH):
    """Distance from the front bumper to the nearest lidar return that sits in
    the corridor around the planned path (metres), or None if the way is free.

    lidar_ego: (N, 4) points in the ego frame. path_ego: (K, 2) planned path in
    the ego frame (first point near the car)."""
    if lidar_ego is None or len(lidar_ego) == 0 or path_ego is None or len(path_ego) < 2:
        return None
    pts = lidar_ego
    m = (pts[:, 2] > AEB_Z_MIN) & (pts[:, 2] < AEB_Z_MAX) & (pts[:, 0] > 0.5)
    m &= ~((np.abs(pts[:, 0]) < front_offset + 0.2) & (np.abs(pts[:, 1]) < 1.3))  # own body
    pts = pts[m][:, 0:2]
    if len(pts) == 0:
        return None
    dense, s = densify(path_ego, 0.25)
    # keep only points that could matter (within the path's bounding box + margin)
    lo = dense.min(axis=0) - half_width
    hi = dense.max(axis=0) + half_width
    pts = pts[(pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
              (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])]
    if len(pts) == 0:
        return None
    d2 = ((pts[:, None, :] - dense[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    lat = np.sqrt(d2[np.arange(len(pts)), nearest])
    inside = lat < half_width
    if not inside.any():
        return None
    return float(max(s[nearest[inside]].min() - front_offset, 0.0))


class AEBMonitor(object):
    """Brakes hard when something solid is inside the swept corridor and the
    car could not stop before it. Works on the lidar only, so it does not rely
    on the simulator's ground truth."""

    def __init__(self, front_offset):
        self.front_offset = front_offset
        self.active = False
        self.activations = 0
        self.last_clearance = None

    @staticmethod
    def required_distance(speed):
        v = max(speed, 0.0)
        return v * v / (2.0 * AEB_DECEL) + AEB_MARGIN + AEB_REACTION * v

    def update(self, lidar_ego, path_ego, speed):
        d = path_clearance(lidar_ego, path_ego, self.front_offset)
        self.last_clearance = d
        need = self.required_distance(speed)
        if d is None:
            hit = False
        elif self.active:
            hit = d < need * AEB_RELEASE
        else:
            hit = d < need
        if hit and not self.active:
            self.activations += 1
        self.active = hit
        return self.active


# ----------------------------------------------------------------------------
# Radar lead vehicle
# ----------------------------------------------------------------------------
class RadarLeadTracker(object):
    """Finds the moving vehicle ahead from radar detections.

    Static things (the parked car, signs, walls) are rejected using the Doppler
    velocity: a static object closes in at exactly our own speed."""

    def __init__(self):
        self.est = None      # [x_ego, y_ego, speed_ground]
        self._lost = 0

    def update(self, radar_ego, ego_speed, path_ego):
        cand = None
        if radar_ego is not None and len(radar_ego):
            x, y, vel = radar_ego[:, 0], radar_ego[:, 1], radar_ego[:, 3]
            ground = ego_speed + vel
            ok = (x > 1.0) & (ground > RADAR_MIN_GROUND_SPEED)
            if path_ego is not None and len(path_ego) >= 2:
                dense, _ = densify(path_ego, 0.5)
                pts = np.stack([x, y], axis=1)
                d = np.sqrt(((pts[:, None, :] - dense[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                ok &= d < RADAR_CORRIDOR
            else:
                ok &= np.abs(y) < 1.8
            if ok.any():
                x0 = x[ok].min()
                grp = ok & (np.abs(x - x0) < 2.5)
                cand = [float(np.median(x[grp])), float(np.median(y[grp])),
                        float(np.median(ground[grp]))]
        if cand is not None:
            if self.est is None:
                self.est = cand
            else:   # light smoothing
                self.est = [0.5 * a + 0.5 * b for a, b in zip(self.est, cand)]
            self._lost = 0
        else:
            self._lost += 1
            if self._lost > RADAR_HOLD_FRAMES:
                self.est = None
        return self.est

    def lead_state_world(self, ego_x, ego_y, ego_yaw):
        """[x, y, speed] in the world frame in the form the planner expects
        (a point near the lead's centre), or None."""
        if self.est is None:
            return None
        w = ego_to_world([[self.est[0] + LEAD_CENTRE_OFFSET, self.est[1]]],
                         ego_x, ego_y, ego_yaw)[0]
        return [float(w[0]), float(w[1]), self.est[2]]


# ----------------------------------------------------------------------------
# Camera projection (front camera, no rotation on the car)
# ----------------------------------------------------------------------------
def camera_intrinsics(width, height, fov_deg):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return f, width / 2.0, height / 2.0


def project_ego_points(pts_ego, width, height, fov_deg):
    """(N, 3) ego-frame points -> (u, v, depth, visible_mask) arrays."""
    f, cx, cy = camera_intrinsics(width, height, fov_deg)
    rel = pts_ego - FRONT_CAM_POS
    depth = rel[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + f * rel[:, 1] / depth
        v = cy - f * rel[:, 2] / depth
    vis = (depth > 0.5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, depth, vis


def box_vertices_ego(ego_xy_yaw, other_xy_yaw, other_z_rel, extent, centre_offset=(0, 0, 0)):
    """The 8 corners of another vehicle's box in the ego frame -> (8, 3).
    other_z_rel: the other car's ground height minus the ego's.
    extent: (half_len, half_wid, half_height). centre_offset: bbox.location."""
    ex, ey, yaw_e = ego_xy_yaw
    ox, oy, yaw_o = other_xy_yaw
    hl, hw, hh = extent
    verts = []
    corners = box_corners(ox, oy, yaw_o, hl, hw, centre_offset[0], centre_offset[1])
    ego_xy = world_to_ego(corners, ex, ey, yaw_e)
    zc = other_z_rel + centre_offset[2]
    for z in (zc - hh, zc + hh):
        for p in ego_xy:
            verts.append([p[0], p[1], z])
    return np.array(verts)


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
