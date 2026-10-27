"""Path generation in route coordinates.

Each candidate keeps the car on the route but shifts it sideways to a target
offset d_t. The shift is a quintic polynomial d(s), so the lateral position,
heading and curvature are all continuous with the car's current state. The
candidates are checked against static obstacles with oriented rectangles and
the cheapest collision free one wins.
"""
from collections import namedtuple

import numpy as np

from route import wrap_angle

TARGET_OFFSETS = np.arange(-3.0, 3.01, 0.5)   # m, positive = right of the route
SAMPLE_DS = 0.5                               # m between path points
CHECK_EVERY = 2                               # collision check every 2nd path point
HARD_MARGIN = 0.30                            # m added around the car for a collision
SOFT_MARGIN = 0.90                            # m: closer than this costs extra

W_OFFSET = 1.0       # prefer staying on the route centre
W_CHANGE = 0.3       # prefer small sideways moves
W_PREV = 0.4         # prefer the previous choice (avoids flip flopping)
W_SOFT = 2.5         # penalty for brushing past an obstacle

Candidate = namedtuple("Candidate", "xy s d target free free_s cost")


def _rect(cx, cy, heading, hl, hw):
    c, s = np.cos(heading), np.sin(heading)
    fwd = np.array([c, s])
    right = np.array([-s, c])
    ctr = np.array([cx, cy])
    return np.array([ctr + fwd * hl + right * hw, ctr + fwd * hl - right * hw,
                     ctr - fwd * hl - right * hw, ctr - fwd * hl + right * hw])


def _overlap(a, b):
    """Separating axis test for two convex quadrilaterals."""
    for poly in (a, b):
        for i in range(2):                    # two edge directions are enough for a quad
            e = poly[i + 1] - poly[i]
            axis = np.array([-e[1], e[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def quintic_offsets(u_len, d0, slope0, d_target, s_rel):
    """d(s) with d(0)=d0, d'(0)=slope0, d''(0)=0 and d=d_target, d'=d''=0 at s=u_len."""
    T = u_len
    A = np.array([[T ** 3, T ** 4, T ** 5],
                  [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                  [6 * T, 12 * T ** 2, 20 * T ** 3]])
    rhs = np.array([d_target - d0 - slope0 * T, -slope0, 0.0])
    a3, a4, a5 = np.linalg.solve(A, rhs)
    x = np.minimum(s_rel, T)
    return d0 + slope0 * x + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5


class PathPlanner(object):
    def __init__(self, route, ego_half_len, ego_half_wid, obstacles):
        self.route = route
        self.hl, self.hw = ego_half_len, ego_half_wid
        self.obstacles = [np.asarray(o, dtype=float) for o in obstacles]
        self.obs_centres = [o.mean(axis=0) for o in self.obstacles]
        self.obs_radius = [np.hypot(*(o - o.mean(axis=0)).T).max() for o in self.obstacles]
        self._prev_target = 0.0

    def _first_hit(self, xy, s, margin):
        """Route position of the first pose that overlaps an obstacle, or None."""
        if not self.obstacles:
            return None
        hl, hw = self.hl + margin, self.hw + margin
        reach = np.hypot(hl, hw)
        for k in range(0, len(xy) - 1, CHECK_EVERY):
            p = xy[k]
            near = [i for i, c in enumerate(self.obs_centres)
                    if np.hypot(*(p - c)) < reach + self.obs_radius[i]]
            if not near:
                continue
            h = np.arctan2(xy[k + 1][1] - p[1], xy[k + 1][0] - p[0])
            rect = _rect(p[0], p[1], h, hl, hw)
            for i in near:
                if _overlap(rect, self.obstacles[i]):
                    return float(s[k])
        return None

    def plan(self, s0, d0, heading_err, speed):
        """Returns (candidates, best_index)."""
        route = self.route
        horizon = float(np.clip(12.0 + 2.5 * speed, 15.0, 35.0))
        horizon = min(horizon, max(route.length - s0, 2.0))
        n = max(int(horizon / SAMPLE_DS) + 1, 3)
        s = s0 + np.arange(n) * SAMPLE_DS
        s_rel = s - s0
        transition = float(np.clip(1.5 * speed + 4.0, 6.0, max(horizon, 6.0)))
        slope0 = float(np.clip(np.tan(wrap_angle(heading_err)), -0.5, 0.5))

        cands = []
        for dt in TARGET_OFFSETS:
            d = quintic_offsets(transition, d0, slope0, dt, s_rel)
            xy = route.to_xy(s, d)
            hit = self._first_hit(xy, s, HARD_MARGIN)
            soft = self._first_hit(xy, s, SOFT_MARGIN) if hit is None else hit
            cost = (W_OFFSET * abs(dt) + W_CHANGE * abs(dt - d0) +
                    W_PREV * abs(dt - self._prev_target))
            if soft is not None and hit is None:
                cost += W_SOFT
            free_s = hit if hit is not None else float(s[-1])
            cands.append(Candidate(xy, s, d, float(dt), hit is None, free_s, cost))

        free = [i for i, c in enumerate(cands) if c.free]
        if free:
            best = min(free, key=lambda i: cands[i].cost)
        else:                               # boxed in: take the one with the most room
            best = max(range(len(cands)), key=lambda i: cands[i].free_s)
        self._prev_target = cands[best].target
        return cands, best
