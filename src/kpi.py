#!/usr/bin/env python3
"""Run metrics for the motion planning demo (the "validation" side).

Everything is computed from the simulator's ground truth, independently of the
planner, so it can be used to judge a run: did it hit anything, how close did it
get, how hard did it brake, where did it stop relative to the line.
"""
import json
import math

import numpy as np


class KPITracker(object):
    def __init__(self, fence, ego_half_len, lead_half_len=2.4, dt=0.05):
        self.fence = fence                      # [x0, y0, x1, y1]
        self.ego_half_len = ego_half_len
        self.lead_half_len = lead_half_len
        self.dt = dt
        self.min_gap = None                     # bumper to bumper, m
        self.min_ttc = None                     # s
        self.max_decel = 0.0
        self.max_accel = 0.0
        self.max_lat_acc = 0.0
        self.max_speed = 0.0
        self.stop_overshoot = None              # m, positive = stopped before the line
        self.stop_time = None
        self.aeb_frames = 0
        self.states = []                        # (t, state name)
        self.gap_now = None
        self.ttc_now = None
        self._lat_acc_hist = []
        self._long_acc_hist = []
        self._stop_captured = False

    # ------------------------------------------------------------------
    def update(self, t, x, y, yaw, speed, accel_xy, lead, planner_state_name,
               stay_stopped, aeb_active):
        """accel_xy: world acceleration (ax, ay) or None.
        lead: None or (x, y, yaw, speed) of the lead car from ground truth."""
        c, s = math.cos(yaw), math.sin(yaw)
        if accel_xy is not None:
            a_long = accel_xy[0] * c + accel_xy[1] * s
            a_lat = -accel_xy[0] * s + accel_xy[1] * c
            # the simulator's acceleration is noisy, average over 0.25 s
            self._long_acc_hist.append(a_long)
            self._lat_acc_hist.append(a_lat)
            n = max(1, int(0.25 / self.dt))
            al = float(np.mean(self._long_acc_hist[-n:]))
            at = float(np.mean(self._lat_acc_hist[-n:]))
            self.max_decel = max(self.max_decel, -al)
            self.max_accel = max(self.max_accel, al)
            self.max_lat_acc = max(self.max_lat_acc, abs(at))
        self.max_speed = max(self.max_speed, speed)
        if aeb_active:
            self.aeb_frames += 1

        self.gap_now = self.ttc_now = None
        if lead is not None:
            lx, ly, lyaw, lspeed = lead
            dx, dy = lx - x, ly - y
            ahead = dx * c + dy * s
            lateral = abs(-dx * s + dy * c)
            if ahead > 0 and lateral < 3.0:
                gap = ahead - self.ego_half_len - self.lead_half_len
                self.gap_now = max(gap, 0.0)
                self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
                closing = speed - lspeed
                if closing > 0.1:
                    self.ttc_now = max(gap, 0.0) / closing
                    self.min_ttc = (self.ttc_now if self.min_ttc is None
                                    else min(self.min_ttc, self.ttc_now))

        if not self.states or self.states[-1][1] != planner_state_name:
            self.states.append((round(t, 2), planner_state_name))

        if stay_stopped and not self._stop_captured and speed < 0.1:
            fx0, fy0, fx1, fy1 = self.fence
            fcx, fcy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
            fx, fy = x + self.ego_half_len * c, y + self.ego_half_len * s   # front bumper
            hx, hy = (fy1 - fy0), -(fx1 - fx0)
            n = math.hypot(hx, hy) or 1.0
            hx, hy = hx / n, hy / n
            if hx * c + hy * s < 0:      # make the lane direction the one we drive in
                hx, hy = -hx, -hy
            self.stop_overshoot = (fcx - fx) * hx + (fcy - fy) * hy
            self.stop_time = t
            self._stop_captured = True

    # ------------------------------------------------------------------
    def report(self, collisions, lane_crossings, reached, duration, path_length,
               has_lead):
        def r(v, n=2):
            return None if v is None else round(float(v), n)
        rep = dict(
            reached_end=bool(reached),
            duration_s=r(duration, 1),
            distance_m=r(path_length, 1),
            collisions=len(collisions),
            collision_with=sorted(set(c[1] for c in collisions)),
            lane_marking_crossings=int(lane_crossings),
            min_gap_to_lead_m=r(self.min_gap) if has_lead else None,
            min_time_to_collision_s=r(self.min_ttc) if has_lead else None,
            stopped_before_line_m=r(self.stop_overshoot),
            max_speed_ms=r(self.max_speed),
            peak_accel_ms2=r(self.max_accel),
            peak_decel_ms2=r(self.max_decel),
            peak_lateral_accel_ms2=r(self.max_lat_acc),
            aeb_active_s=r(self.aeb_frames * self.dt, 2),
            planner_states=self.states,
        )
        return rep


def verdict(rep):
    """Simple pass/fail summary for the report."""
    problems = []
    if not rep["reached_end"]:
        problems.append("did not reach the end of the route")
    if rep["collisions"]:
        problems.append("%d collision(s)" % rep["collisions"])
    if rep["min_gap_to_lead_m"] is not None and rep["min_gap_to_lead_m"] < 1.0:
        problems.append("closer than 1 m to the lead car")
    if rep["stopped_before_line_m"] is not None and rep["stopped_before_line_m"] < -0.3:
        problems.append("stopped past the stop line")
    return ("PASS", []) if not problems else ("FAIL", problems)


def print_report(rep):
    status, problems = verdict(rep)
    print("")
    print("---- run report: %s %s----" % (status, ("(" + "; ".join(problems) + ") ") if problems else ""))
    rows = [("reached the end", rep["reached_end"]),
            ("duration (s)", rep["duration_s"]),
            ("distance (m)", rep["distance_m"]),
            ("collisions", rep["collisions"]),
            ("lane marking crossings", rep["lane_marking_crossings"]),
            ("min gap to lead (m)", rep["min_gap_to_lead_m"]),
            ("min time to collision (s)", rep["min_time_to_collision_s"]),
            ("stopped before the line (m)", rep["stopped_before_line_m"]),
            ("max speed (m/s)", rep["max_speed_ms"]),
            ("peak accel / decel (m/s2)", "%s / %s" % (rep["peak_accel_ms2"], rep["peak_decel_ms2"])),
            ("peak lateral accel (m/s2)", rep["peak_lateral_accel_ms2"]),
            ("emergency braking time (s)", rep["aeb_active_s"])]
    for k, v in rows:
        print("  %-30s %s" % (k, v))


def save_report(rep, path):
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2)
