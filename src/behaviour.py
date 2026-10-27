"""High level decision making: cruise, stop at a line, follow a lead car.

The behaviour layer does not plan any geometry. It only decides which
constraints the path and speed planners must respect right now.
"""
from collections import namedtuple

CRUISE = 0
APPROACH_STOP = 1
HOLD_AT_LINE = 2

STATE_NAMES = {CRUISE: "CRUISE", APPROACH_STOP: "APPROACH_STOP",
               HOLD_AT_LINE: "HOLD_AT_LINE"}

# What the lower layers receive.
#   stop_s    : route position where the car's reference point must come to rest (or None)
#   lead      : (gap_m, lead_speed) of the vehicle we follow (or None)
Decision = namedtuple("Decision", "state stop_s lead")

COMFORT_DECEL = 1.5       # m/s^2 used to decide when to start braking for the line
STOP_BUFFER = 1.0         # m between the front bumper and the line when stopped
HOLD_SECONDS = 3.0        # time spent waiting at the line
FOLLOW_RANGE = 30.0       # m, look this far ahead for a lead vehicle
LEAD_LANE_HALF_WIDTH = 2.2  # m, a vehicle this close to the route centre counts as in our lane


class Behaviour(object):
    def __init__(self, route, stop_s_list, ego_half_len, lead_half_len=2.4):
        self.route = route
        self.stop_s_list = sorted(stop_s_list)
        self.ego_half_len = ego_half_len
        self.lead_half_len = lead_half_len
        self.state = CRUISE
        self._hold_started = None
        self._done = set()          # stop lines already served
        self._active = None         # stop line currently being approached

    def _stop_target(self, stop_s):
        return stop_s - self.ego_half_len - STOP_BUFFER

    def update(self, s_ego, speed, t, lead_xy_speed=None):
        # ---- stop line state machine ----
        if self.state == CRUISE:
            for stop_s in self.stop_s_list:
                if stop_s in self._done or stop_s < s_ego:
                    continue
                dist = self._stop_target(stop_s) - s_ego
                braking = speed * speed / (2.0 * COMFORT_DECEL)
                if dist < braking + 8.0:
                    self._active = stop_s
                    self.state = APPROACH_STOP
                break
        elif self.state == APPROACH_STOP:
            dist = self._stop_target(self._active) - s_ego
            if speed < 0.1 and dist < 2.0:
                self.state = HOLD_AT_LINE
                self._hold_started = t
        elif self.state == HOLD_AT_LINE:
            if t - self._hold_started >= HOLD_SECONDS:
                self._done.add(self._active)
                self._active = None
                self.state = CRUISE

        stop_s = self._stop_target(self._active) if self._active is not None else None

        # ---- lead vehicle ----
        lead = None
        if lead_xy_speed is not None:
            ls, ld, _ = self.route.project(lead_xy_speed[0], lead_xy_speed[1])
            ahead = ls - s_ego
            if 0.0 < ahead < FOLLOW_RANGE and abs(ld) < LEAD_LANE_HALF_WIDTH:
                gap = ahead - self.ego_half_len - self.lead_half_len
                lead = (max(gap, 0.0), lead_xy_speed[2])
        return Decision(self.state, stop_s, lead)
