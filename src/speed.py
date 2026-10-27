"""Speed profile along the chosen path.

Upper bounds on speed come from the route limit, path curvature, a stop
point and a lead vehicle. A backward pass then limits deceleration and a
forward pass limits acceleration, so the profile is always drivable.
"""
import numpy as np

ACCEL = 1.5            # m/s^2 comfortable acceleration
DECEL = 1.8            # m/s^2 comfortable braking
LATERAL_ACCEL = 3.0    # m/s^2 allowed in curves
MIN_GAP = 4.0          # m standstill gap behind a lead vehicle
TIME_GAP = 1.2         # s headway behind a lead vehicle
GAP_GAIN = 0.6         # 1/s, how hard the gap error changes the target speed


def path_curvature(xy):
    d = np.diff(xy, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    heading = np.append(heading, heading[-1])
    ds = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-3)
    ds = np.append(ds, ds[-1])
    return np.abs(np.gradient(heading) / ds)


def plan_speed(xy, s, route, v0, stop_s=None, lead=None, hold=False, blocked_s=None):
    """Speed for every point of the path.

    xy, s    : path points and their route position
    v0       : current speed
    stop_s   : route position where the car must stop (stop line), or None
    lead     : (gap, lead_speed) or None
    hold     : True while waiting at the line
    blocked_s: route position of an obstacle we cannot pass, or None
    """
    n = len(s)
    if hold:
        return np.zeros(n)
    p = s - s[0]                                    # distance along the path
    step = np.maximum(np.diff(p), 1e-3)

    limit = np.array([route.limit_at(v) for v in s])
    kappa = path_curvature(xy)
    target = np.minimum(limit, np.sqrt(LATERAL_ACCEL / np.maximum(kappa, 1e-3)))

    for stop in (stop_s, blocked_s):
        if stop is not None:
            remaining = np.maximum(stop - s, 0.0)
            target = np.minimum(target, np.sqrt(2.0 * DECEL * remaining))
    if lead is not None:
        gap, vl = lead
        wanted_gap = MIN_GAP + TIME_GAP * max(v0, 0.0)
        cap = max(vl + GAP_GAIN * (gap - wanted_gap), 0.0)
        target = np.minimum(target, cap)

    v = target.copy()
    for i in range(n - 2, -1, -1):                  # braking limit, from the end backwards
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * DECEL * step[i]))
    v[0] = min(v[0], max(v0, 0.0))                  # start from the real speed
    for i in range(n - 1):                          # acceleration limit, forwards
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * ACCEL * step[i]))
    return v
