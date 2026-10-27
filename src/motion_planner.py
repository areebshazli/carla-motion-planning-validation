"""Planning pipeline: behaviour -> path -> speed, behind one update() call.

Nothing in here imports carla, so the same code runs in the CARLA client and
in the offline test (offline_sim.py).
"""
import math

import numpy as np

import behaviour
from behaviour import Behaviour
from paths import PathPlanner
from route import Route, wrap_angle
from speed import plan_speed


def interpolate_waypoints(waypoints, resolution=0.1):
    """Insert points so consecutive waypoints are at most `resolution` apart."""
    wps = np.asarray(waypoints, dtype=float)
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = int(length / resolution)
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / float(n + 1)))
        out.append(b)
    return np.array(out).tolist()


def stop_line_across_lane(cx, cy, heading, length=5.0):
    """Stop line [x0, y0, x1, y1] centred on (cx, cy), perpendicular to the lane."""
    nx, ny = -math.sin(heading), math.cos(heading)
    h = length / 2.0
    return [cx - h * nx, cy - h * ny, cx + h * nx, cy + h * ny]


class MotionPlanner(object):
    """waypoints : [[x, y, speed_limit], ...] global route
    stop_lines: [[x0, y0, x1, y1], ...] lines the car has to stop at once
    obstacles : list of polygons (4 corner points each) of static obstacles
    """

    def __init__(self, waypoints, stop_lines, obstacles, ego_half_len=2.45, ego_half_wid=0.95):
        self.route = Route(waypoints, smooth=True)
        stop_s = []
        for x0, y0, x1, y1 in stop_lines:
            stop_s.append(self.route.project(0.5 * (x0 + x1), 0.5 * (y0 + y1))[0])
        self.behaviour = Behaviour(self.route, stop_s, ego_half_len)
        self.paths = PathPlanner(self.route, ego_half_len, ego_half_wid, obstacles)
        self.local_waypoints = None
        self.debug = {}

    @property
    def state(self):
        return self.behaviour.state

    def update(self, x, y, yaw, speed, t, lead_state=None):
        """One planning cycle. lead_state: None or [x, y, speed] of the car ahead.
        Returns the local path as [[x, y, speed], ...]."""
        s0, d0, route_heading = self.route.project(x, y)
        decision = self.behaviour.update(s0, speed, t, lead_state)

        cands, best = self.paths.plan(s0, d0, wrap_angle(yaw - route_heading), speed)
        chosen = cands[best]
        blocked_s = None
        if not chosen.free:
            # nothing collision free: stop before the first obstacle on the best effort path
            blocked_s = chosen.free_s - self.paths.hl - 1.5

        v = plan_speed(chosen.xy, chosen.s, self.route, speed, stop_s=decision.stop_s,
                       lead=decision.lead, hold=(decision.state == behaviour.HOLD_AT_LINE),
                       blocked_s=blocked_s)
        self.local_waypoints = np.column_stack([chosen.xy, v]).tolist()
        self.debug = dict(paths=[c.xy.T for c in cands], best_index=best,
                          collision_check_array=np.array([c.free for c in cands]),
                          decision=decision, route_s=s0, route_d=d0)
        return self.local_waypoints
