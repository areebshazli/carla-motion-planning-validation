"""A synthetic road used by offline_sim.py: a long straight, a stop line, then
a right turn. Same layout the CARLA client builds from the real map."""
import math

import numpy as np

CRUISE = 8.3          # m/s
TURN_SPEED = 2.5      # m/s
STRAIGHT = 220.0      # m before the turn
RADIUS = 8.0          # m
TAIL = 40.0           # m of road after the turn
SPACING = 2.0


def build(parked_s=60.0, parked_offset=1.2):
    """Returns waypoints [[x, y, v]], stop line [x0, y0, x1, y1], parked car polygon."""
    pts = []
    for s in np.arange(0.0, STRAIGHT, SPACING):
        pts.append((s, 0.0, 0.0))
    arc = RADIUS * math.pi / 2
    for s in np.arange(0.0, arc, SPACING):
        a = s / RADIUS
        pts.append((STRAIGHT + RADIUS * math.sin(a), RADIUS - RADIUS * math.cos(a), math.degrees(a)))
    for s in np.arange(0.0, TAIL + SPACING, SPACING):
        pts.append((STRAIGHT + RADIUS, RADIUS + s, 90.0))
    wps = []
    for i, (x, y, yaw) in enumerate(pts):
        slow = STRAIGHT - 10.0 <= i * SPACING
        wps.append([x, y, TURN_SPEED if slow else CRUISE])
    stop_x = STRAIGHT - 2.0
    stop_line = [stop_x, -2.5, stop_x, 2.5]
    px = parked_s
    py = parked_offset
    parked = [[px + 2.45, py + 0.95], [px + 2.45, py - 0.95],
              [px - 2.45, py - 0.95], [px - 2.45, py + 0.95]]
    return wps, stop_line, parked
