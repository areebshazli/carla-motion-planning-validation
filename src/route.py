"""Global route as a polyline with arclength, so everything else can work in
route coordinates: s = distance along the route, d = lateral offset
(positive to the right, the same sense as CARLA's y axis).
"""
import math

import numpy as np


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def smooth_waypoints(waypoints, spacing=0.5, window=9):
    """Resample a coarse route to `spacing` metres and round off the corners
    with a moving average, so heading and curvature vary smoothly."""
    wp = np.asarray(waypoints, dtype=float)
    seg = np.hypot(*np.diff(wp[:, 0:2], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, s[-1], spacing)
    ss = np.append(ss, s[-1])
    out = np.stack([np.interp(ss, s, wp[:, k]) for k in range(wp.shape[1])], axis=1)
    pad = window // 2
    kernel = np.ones(window) / window
    for k in (0, 1):
        # extend the ends in a straight line so the average does not pull them inwards
        head = out[0, k] - (out[1:pad + 1, k] - out[0, k])[::-1]
        tail = out[-1, k] + (out[-1, k] - out[-pad - 1:-1, k])[::-1]
        padded = np.concatenate([head, out[:, k], tail])
        out[:, k] = np.convolve(padded, kernel, mode="valid")
    return out


class Route(object):
    def __init__(self, waypoints, smooth=False):
        """waypoints: [[x, y, speed_limit], ...] in world coordinates."""
        wp = smooth_waypoints(waypoints) if smooth else np.asarray(waypoints, dtype=float)
        self.xy = wp[:, 0:2]
        self.limit = wp[:, 2] if wp.shape[1] > 2 else np.full(len(wp), 10.0)
        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
        # heading at each vertex: average of the two neighbouring segments
        vh = np.empty(len(self.xy))
        vh[0], vh[-1] = seg_heading[0], seg_heading[-1]
        for i in range(1, len(self.xy) - 1):
            a, b = seg_heading[i - 1], seg_heading[i]
            vh[i] = a + 0.5 * wrap_angle(b - a)
        self.heading_v = np.unwrap(vh)
        self._last = 0

    # ------------------------------------------------------------------
    def project(self, x, y):
        """Nearest point on the route -> (s, d, heading)."""
        p = np.array([x, y])
        a = self.xy[:-1]
        ab = self.xy[1:] - a
        L2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
        t = np.clip(((p - a) * ab).sum(axis=1) / L2, 0.0, 1.0)
        foot = a + ab * t[:, None]
        dist2 = ((foot - p) ** 2).sum(axis=1)
        i = int(dist2.argmin())
        s = self.s[i] + t[i] * self.seg_len[i]
        hd = self.heading_at(s)
        # signed lateral offset: positive when p is to the right of the route
        rx, ry = -math.sin(hd), math.cos(hd)
        d = (p[0] - foot[i][0]) * rx + (p[1] - foot[i][1]) * ry
        return float(s), float(d), float(hd)

    def _index(self, s):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.s, s, side="right")) - 1
        return min(max(i, 0), len(self.seg_len) - 1)

    def heading_at(self, s):
        i = self._index(s)
        f = (s - self.s[i]) / max(self.seg_len[i], 1e-9)
        f = float(np.clip(f, 0.0, 1.0))
        return float(self.heading_v[i] + f * (self.heading_v[i + 1] - self.heading_v[i]))

    def point_at(self, s):
        i = self._index(s)
        f = float(np.clip((s - self.s[i]) / max(self.seg_len[i], 1e-9), 0.0, 1.0))
        p = self.xy[i] + f * (self.xy[i + 1] - self.xy[i])
        return float(p[0]), float(p[1])

    def limit_at(self, s):
        return float(np.interp(s, self.s, self.limit))

    def to_xy(self, s_arr, d_arr):
        """Route coordinates -> world points (K, 2)."""
        s_arr = np.clip(np.asarray(s_arr, dtype=float), 0.0, self.length)
        d_arr = np.asarray(d_arr, dtype=float)
        px = np.interp(s_arr, self.s, self.xy[:, 0])
        py = np.interp(s_arr, self.s, self.xy[:, 1])
        h = np.interp(s_arr, self.s, self.heading_v)
        return np.stack([px - d_arr * np.sin(h), py + d_arr * np.cos(h)], axis=1)
