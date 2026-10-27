#!/usr/bin/env python3
"""Closed loop test of the planner and controller without CARLA.

The car is a kinematic bicycle model with steering lag and simple drive and
brake dynamics. The scenario is scenario.py: a parked car partly blocking the
lane, a slow lead car, a stop line and a right turn.

    python offline_sim.py
    python offline_sim.py --no-lead --lead-speed 3.0 --out result.png
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import behaviour
import control
import motion_planner as mp
import scenario

DT = 0.05
PLAN_EVERY = 2
WAIT = 1.0
WHEELBASE = 2.9
MAX_STEER_DEG = 70.0
CAR_LEN, CAR_WID = 4.9, 1.9


def corners(x, y, yaw, hl, hw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return np.array([[x + a * c - b * s, y + a * s + b * c] for a, b in pts])


def overlap(r1, r2):
    for poly in (r1, r2):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-e[1], e[0]])
            p1, p2 = r1 @ axis, r2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lead", action="store_true")
    ap.add_argument("--no-parked", action="store_true")
    ap.add_argument("--lead-speed", type=float, default=3.5)
    ap.add_argument("--lead-s", type=float, default=90.0, help="where the lead car starts along the road")
    ap.add_argument("--out", default=os.path.join(HERE, "offline_sim.png"))
    args = ap.parse_args()

    wps, stop_line, parked = scenario.build()
    obstacles = [] if args.no_parked else [parked]
    planner = mp.MotionPlanner(wps, [stop_line], obstacles, CAR_LEN / 2, CAR_WID / 2)
    ctrl = control.PathController(max_steer_deg=MAX_STEER_DEG)
    route = planner.route

    x, y = wps[0][0], wps[0][1]
    yaw = route.heading_at(0.0)
    v, steer_act = 0.0, 0.0
    lead_s = args.lead_s

    log = {k: [] for k in ("t", "x", "y", "v", "vref", "thr", "brk", "st")}
    collided, reached = False, False
    min_lead_gap, min_parked = 1e9, 1e9
    states, stop_info = [], None
    end = np.array(wps[-1][:2])
    for k in range(int(150.0 / DT)):
        t = k * DT
        if t >= WAIT and not args.no_lead:
            lead_s += args.lead_speed * DT
        lx, ly = route.point_at(lead_s)
        lyaw = route.heading_at(lead_s)
        lead_gone = lead_s > route.length - 4.0     # it drives off the end of our test road
        lead_state = None if (args.no_lead or lead_gone) else [lx, ly, args.lead_speed]

        if t < WAIT:
            thr, st, brk = 0.0, 0.0, 1.0
        else:
            if k % PLAN_EVERY == 0:
                ctrl.set_path(planner.update(x, y, yaw, v, t - WAIT, lead_state))
            thr, st, brk = ctrl.step(x, y, yaw, v, DT)

        steer_act += (st - steer_act) * min(1.0, DT / 0.2)
        delta = steer_act * math.radians(MAX_STEER_DEG)
        accel = 3.2 * thr - 0.0006 * v * v - (0.25 if v > 0.01 else 0.0) - 7.0 * brk * (1 if v > 0.01 else 0)
        v = max(0.0, v + accel * DT)
        yaw += v / WHEELBASE * math.tan(delta) * DT
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT

        ego = corners(x, y, yaw, CAR_LEN / 2, CAR_WID / 2)
        if not args.no_parked:
            min_parked = min(min_parked, min(np.hypot(*(np.array(parked) - [x, y]).T)))
            if overlap(ego, np.array(parked)):
                collided = True
        if lead_state is not None:
            if overlap(ego, corners(lx, ly, lyaw, CAR_LEN / 2, CAR_WID / 2)):
                collided = True
            ahead = (lx - x) * math.cos(yaw) + (ly - y) * math.sin(yaw)
            if ahead > 0:
                min_lead_gap = min(min_lead_gap, ahead - CAR_LEN)
        name = behaviour.STATE_NAMES[planner.state]
        if not states or states[-1][1] != name:
            states.append((round(t, 1), name))
        if planner.state == behaviour.HOLD_AT_LINE and stop_info is None and v < 0.1:
            front = x + CAR_LEN / 2 * math.cos(yaw)
            stop_info = (stop_line[0] - front, t)

        for key, val in zip(log, (t, x, y, v, ctrl.desired_speed, thr, brk, st)):
            log[key].append(val)
        if math.hypot(end[0] - x, end[1] - y) < 2.0:
            reached = True
            break

    wp = np.array(wps)
    print("reached the end       : %s (t=%.1f s)" % (reached, log["t"][-1]))
    print("collision             : %s" % collided)
    if not args.no_parked:
        print("closest to parked car (centre of nearest corner): %.2f m" % min_parked)
    if not args.no_lead:
        print("closest gap to lead   : %.2f m" % min_lead_gap)
    print("behaviour states      : %s" % states)
    if stop_info:
        print("stopped %.2f m before the line at t=%.1f s" % stop_info)
    print("top speed             : %.2f m/s" % max(log["v"]))
    print("max steer command     : %.2f" % max(abs(np.array(log["st"]))))

    fig, ax = plt.subplots(3, 1, figsize=(10, 11))
    ax[0].plot(wp[:, 0], wp[:, 1], "g-", label="route")
    ax[0].plot(log["x"], log["y"], color="orange", label="car")
    ax[0].plot([stop_line[0], stop_line[2]], [stop_line[1], stop_line[3]], "r-", lw=3, label="stop line")
    if not args.no_parked:
        ax[0].add_patch(plt.Polygon(parked, color="b", alpha=0.4, label="parked car"))
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].invert_yaxis()
    ax[0].set_title("Top down")
    ax[1].plot(log["t"], log["v"], label="speed"); ax[1].plot(log["t"], log["vref"], "--", label="target")
    ax[1].legend(); ax[1].grid(True); ax[1].set_ylabel("m/s")
    ax[2].plot(log["t"], log["thr"], label="throttle"); ax[2].plot(log["t"], log["brk"], label="brake")
    ax[2].plot(log["t"], log["st"], label="steer"); ax[2].legend(); ax[2].grid(True); ax[2].set_xlabel("s")
    fig.tight_layout(); fig.savefig(args.out, dpi=90)
    print("saved", args.out)
    return 0 if (reached and not collided) else 1


if __name__ == "__main__":
    sys.exit(main())
