#!/usr/bin/env python3
"""Motion planning demo for CARLA 0.9.x.

A car drives a lane towards an intersection. On the way it swerves around a
parked car, follows a slower lead car, stops at a line, waits, and turns.
The planner (behaviour -> lane-offset paths -> speed profile) and the
Stanley / PI controller run in synchronous mode at 20 Hz. Optional lidar,
radar and camera sensors feed a safety brake, a lead car tracker and a
dashboard video, and every run produces a metrics report.

Start CARLA (Unreal Editor, press Play), then in this folder:

    python carla_motion_planning.py
    python carla_motion_planning.py --dashboard --aeb --radar-lead
    python carla_motion_planning.py --headless --no-lead --turn left

Ctrl+C stops cleanly: the simulator is put back into normal mode and the
spawned actors are removed.
"""
import argparse
import glob
import itertools
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

try:
    import carla
except ImportError:
    sys.exit(
        "Could not import the carla module.\n"
        "Install the wheel you built with `make PythonAPI`:\n"
        "    pip install <carla_root>\\PythonAPI\\carla\\dist\\carla-*.whl\n"
        "into the same Python (3.12) you run this script with.")

import behaviour
import control
import kpi
import motion_planner as mp
import perception as pc
import sensor_rig
from behaviour import HOLD_AT_LINE, STATE_NAMES

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FIXED_DELTA_SECONDS = 0.05   # simulation step (20 Hz)
WAIT_TIME_BEFORE_START = 1.0 # s, cars held still after spawning
LP_FREQUENCY_DIVISOR = 2     # planner runs every 2nd simulation step
WP_SPACING = 2.0             # m between global waypoints
STOP_FENCE_LENGTH = 5.0      # m, virtual stop line width
SLOW_ZONE_BEFORE_JUNCTION = 10.0  # m before the junction where speed drops
TURN_SPEED = 2.5             # m/s through the intersection
PARKED_LATERAL_OFFSET = 1.2  # m the parked car sits off the lane centre (right)
OUTPUT_DIR = os.path.join(HERE, "controller_output")

EGO_BLUEPRINTS = ["vehicle.lincoln.mkz_2020", "vehicle.lincoln.mkz_2017",
                  "vehicle.tesla.model3"]
PARKED_BLUEPRINTS = ["vehicle.audi.etron", "vehicle.audi.a2",
                     "vehicle.tesla.model3"]
LEAD_BLUEPRINTS = ["vehicle.tesla.model3", "vehicle.audi.a2",
                   "vehicle.lincoln.mkz_2017"]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def pick_blueprint(bp_lib, candidates, role):
    for name in candidates:
        found = bp_lib.filter(name)
        if len(found) > 0:
            bp = found[0]
            break
    else:  # none of the preferred ones exist in this build
        cars = [b for b in bp_lib.filter("vehicle.*")
                if b.has_attribute("number_of_wheels") and
                int(b.get_attribute("number_of_wheels")) == 4]
        bp = cars[0]
    bp.set_attribute("role_name", role)
    return bp


def lerp_loc(a, b, f):
    return carla.Location(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f,
                          a.z + (b.z - a.z) * f)


def yaw_diff_deg(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def vehicle_state(actor):
    """x, y, yaw (rad), forward speed (m/s)."""
    tr = actor.get_transform()
    vel = actor.get_velocity()
    yaw = math.radians(tr.rotation.yaw)
    speed = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
    return tr.location.x, tr.location.y, yaw, speed


def box_points(actor):
    """Border points [x, y] of a vehicle bounding box in the world frame.

    Corners and edge midpoints, eight points per car.
    """
    bb = actor.bounding_box
    ex, ey = bb.extent.x, bb.extent.y
    local = [(-ex, -ey), (-ex, 0), (-ex, ey), (0, ey),
             (ex, ey), (ex, 0), (ex, -ey), (0, -ey)]
    tr = actor.get_transform()
    yaw = math.radians(tr.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for lx, ly in local:
        lx += bb.location.x
        ly += bb.location.y
        pts.append([tr.location.x + lx * c - ly * s,
                    tr.location.y + lx * s + ly * c])
    return pts


# ----------------------------------------------------------------------------
# Scenario construction from the CARLA map
# ----------------------------------------------------------------------------
def build_route(start_wp, max_length, spacing, turn):
    """Follow the lane from start_wp. At forks choose by `turn`."""
    route = [start_wp]
    travelled = 0.0
    while travelled < max_length:
        nxt = route[-1].next(spacing)
        if not nxt:
            break
        if len(nxt) > 1:
            nxt = [choose_branch(route[-1], nxt, turn)]
        route.append(nxt[0])
        travelled += spacing
    return route


def choose_branch(current, candidates, turn):
    """Pick a branch. CARLA yaw is clockwise positive, so right turns have a
    positive yaw change."""
    scored = []
    for c in candidates:
        ahead = c.next(12.0)
        ref = ahead[0] if ahead else c
        scored.append((yaw_diff_deg(current.transform.rotation.yaw,
                                    ref.transform.rotation.yaw), c))
    if turn == "right":
        return max(scored, key=lambda s: s[0])[1]
    if turn == "left":
        return min(scored, key=lambda s: s[0])[1]
    if turn == "straight":
        return min(scored, key=lambda s: abs(s[0]))[1]
    return scored[0][1]


def junction_span(route):
    """(first junction index, first index after it) or (None, None)."""
    j = next((i for i, w in enumerate(route) if w.is_junction), None)
    if j is None:
        return None, None
    k = next((i for i in range(j, len(route)) if not route[i].is_junction),
             len(route) - 1)
    return j, k


def find_starts(world_map, spacing, turn, min_dist=120.0, max_dist=260.0):
    """Spawn points whose lane runs into a junction, best candidates first.

    Returns [(spawn_index, route), ...]: first the ones with the junction
    120 to 260 m ahead, then any others with at least 60 m of road before it.
    """
    good, fallback = [], []
    for idx, sp in enumerate(world_map.get_spawn_points()):
        wp = world_map.get_waypoint(sp.location)
        route = build_route(wp, 450.0, spacing, turn)
        j, k = junction_span(route)
        if j is None:
            continue
        d = j * spacing
        tail_ok = len(route) - k >= int(25.0 / spacing)
        if not tail_ok:
            continue
        if min_dist <= d <= max_dist:
            good.append((idx, route))
        elif d > 60.0:
            fallback.append((idx, route))
    return good + fallback


def make_waypoints(route, cruise_speed, spacing):
    """[[x, y, v], ...] with a reduced speed around the junction, plus the
    stop line [x0, y0, x1, y1] just before it."""
    j, k = junction_span(route)
    keep = k + int(25.0 / spacing)
    route = route[:keep + 1]
    slow_start = max(0, j - int(SLOW_ZONE_BEFORE_JUNCTION / spacing))
    waypoints = []
    for i, w in enumerate(route):
        v = TURN_SPEED if slow_start <= i <= k + 1 else cruise_speed
        waypoints.append([w.transform.location.x, w.transform.location.y, v])
    a, b = route[j - 2].transform.location, route[j - 1].transform.location
    mid = lerp_loc(a, b, 0.5)
    heading = math.atan2(b.y - a.y, b.x - a.x)
    fence = mp.stop_line_across_lane(mid.x, mid.y, heading, STOP_FENCE_LENGTH)
    return route, waypoints, fence, j


def box_polygon(actor):
    """The four ground corners of a vehicle's bounding box in the world frame."""
    bb = actor.bounding_box
    tr = actor.get_transform()
    return pc.box_corners(tr.location.x, tr.location.y, math.radians(tr.rotation.yaw),
                          bb.extent.x, bb.extent.y, bb.location.x, bb.location.y)


def spawn_parked_car(world, bp_lib, route, index, blueprints=None):
    wp = route[index]
    right = wp.transform.get_right_vector()
    bp = pick_blueprint(bp_lib, blueprints or PARKED_BLUEPRINTS, "parked")
    for offset in (PARKED_LATERAL_OFFSET, 0.9, 0.6, 0.3):
        loc = carla.Location(wp.transform.location.x + right.x * offset,
                             wp.transform.location.y + right.y * offset,
                             wp.transform.location.z + 0.4)
        car = world.try_spawn_actor(bp, carla.Transform(loc, wp.transform.rotation))
        if car is not None:
            car.apply_control(carla.VehicleControl(hand_brake=True))
            return car
    return None


def spawn_on_route(world, bp_lib, wp, blueprints, role):
    bp = pick_blueprint(bp_lib, blueprints, role)
    for dz in (0.4, 1.0, 2.0):   # a little higher if the first try collides
        loc = carla.Location(wp.transform.location.x, wp.transform.location.y,
                             wp.transform.location.z + dz)
        actor = world.try_spawn_actor(bp, carla.Transform(loc, wp.transform.rotation))
        if actor is not None:
            return actor
    return None


def max_steer_degrees(vehicle, default=70.0):
    try:
        wheels = vehicle.get_physics_control().wheels
        return max(w.max_steer_angle for w in wheels) or default
    except Exception:
        return default


def update_spectator(spectator, mode, ego, x, y, yaw, frame):
    """Move the Editor camera: 'overhead', 'chase' or 'free' (do not touch it)."""
    z = ego.get_transform().location.z
    if mode == "overhead" and frame % 4 == 0:
        spectator.set_transform(carla.Transform(
            carla.Location(x, y, z + 45.0), carla.Rotation(pitch=-90)))
    elif mode == "chase":
        spectator.set_transform(carla.Transform(
            carla.Location(x - 9.0 * math.cos(yaw), y - 9.0 * math.sin(yaw), z + 4.5),
            carla.Rotation(pitch=-14.0, yaw=math.degrees(yaw))))


def encode_video(frames_dir, fps, out_path):
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frames:
        print("No frames were recorded.")
        return
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(imageio.imread(f))
        writer.close()
        print("Video saved:", out_path)
    except Exception as exc:
        print("Recorded %d frames in %s" % (len(frames), frames_dir))
        print("Could not make the video automatically (%s)." % exc)
        print("Either run  pip install imageio imageio-ffmpeg  and run the script again,")
        print("or, with ffmpeg installed, in that folder run:")
        print("  ffmpeg -framerate %d -i %%06d.png -c:v libx264 -pix_fmt yuv420p run.mp4" % fps)


class VideoSink(object):
    """Writes dashboard frames straight to an MP4. If imageio / ffmpeg are
    missing it falls back to numbered PNGs (make_video.py can encode those)."""

    def __init__(self, out_path, frames_dir, fps):
        self.out_path, self.frames_dir, self.fps = out_path, frames_dir, fps
        self.writer, self.count, self.png = None, 0, False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                             quality=8, macro_block_size=1)
        except Exception as exc:
            print("Could not start the video encoder (%s)." % exc)
            print("Saving PNG frames instead; later run  pip install imageio imageio-ffmpeg  "
                  "and  python make_video.py --frames %s" % frames_dir)
            self._png_mode()

    def _png_mode(self):
        self.png = True
        self.writer = None
        os.makedirs(self.frames_dir, exist_ok=True)
        for old in glob.glob(os.path.join(self.frames_dir, "*.png")):
            os.remove(old)

    def add(self, frame):
        if self.png:
            from PIL import Image
            Image.fromarray(frame).save(os.path.join(self.frames_dir, "%06d.png" % self.count))
        else:
            self.writer.append_data(frame)
        self.count += 1

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
                print("Video saved:", self.out_path)
            except Exception as exc:
                print("Could not finish the video: %s" % exc)
        elif self.png and self.count:
            print("Saved %d PNG frames in %s" % (self.count, self.frames_dir))


def actor_box(actor, ego_pose, ego_z, colour, label):
    """Bounding box of another car as the dashboard wants it."""
    ex, ey, eyaw = ego_pose
    tr = actor.get_transform()
    bb = actor.bounding_box
    yaw = math.radians(tr.rotation.yaw)
    ext = (bb.extent.x, bb.extent.y, bb.extent.z)
    corners_w = pc.box_corners(tr.location.x, tr.location.y, yaw, bb.extent.x, bb.extent.y,
                               bb.location.x, bb.location.y)
    poly = pc.world_to_ego(corners_w, ex, ey, eyaw)
    verts = pc.box_vertices_ego(ego_pose, (tr.location.x, tr.location.y, yaw),
                                tr.location.z - ego_z, ext,
                                (bb.location.x, bb.location.y, bb.location.z))
    return dict(poly=poly, verts=verts, colour=colour, label=label)


def send(actor, throttle, steer, brake, hand_brake=False):
    actor.apply_control(carla.VehicleControl(
        throttle=float(np.clip(throttle, 0.0, 1.0)),
        steer=float(np.clip(steer, -1.0, 1.0)),
        brake=float(np.clip(brake, 0.0, 1.0)),
        hand_brake=hand_brake))


# ----------------------------------------------------------------------------
# Drawing in the simulator window (helps a lot when watching in the Editor)
# ----------------------------------------------------------------------------
def draw_scene(world, waypoints, fence, box_pts):
    dbg = world.debug
    z = 0.6
    try:
        for x, y, v in waypoints:
            dbg.draw_point(carla.Location(x, y, z), size=0.08,
                           color=carla.Color(0, 200, 0), life_time=600.0)
        dbg.draw_line(carla.Location(fence[0], fence[1], z),
                      carla.Location(fence[2], fence[3], z),
                      thickness=0.15, color=carla.Color(255, 0, 0),
                      life_time=600.0)
        for x, y in box_pts:
            dbg.draw_point(carla.Location(x, y, z), size=0.1,
                           color=carla.Color(0, 80, 255), life_time=600.0)
    except Exception as exc:  # drawing is optional
        print("(debug drawing unavailable: %s)" % exc)


def draw_plan(world, planner):
    try:
        dbg = world.debug
        for p, path in enumerate(planner.debug.get("paths", [])):
            best = (p == planner.debug.get("best_index"))
            col = carla.Color(255, 140, 0) if best else carla.Color(60, 60, 255)
            for i in range(0, len(path[0]), 5):
                dbg.draw_point(carla.Location(path[0][i], path[1][i], 0.7),
                               size=0.06 if not best else 0.1, color=col,
                               life_time=2 * FIXED_DELTA_SECONDS * LP_FREQUENCY_DIVISOR)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Optional live plot
# ----------------------------------------------------------------------------
class LivePlot(object):
    def __init__(self, waypoints, fence, box_pts):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(9, 8))
        wp = np.array(waypoints)
        self.ax.plot(wp[:, 0], wp[:, 1], "g-")
        self.ax.plot([fence[0], fence[2]], [fence[1], fence[3]], "r-", lw=3)
        bp = np.array(box_pts)
        self.ax.plot(bp[:, 0], bp[:, 1], "b+")
        self.trace, = self.ax.plot([], [], color=(1, .5, 0))
        self.car, = self.ax.plot([], [], "bs")
        self.lead, = self.ax.plot([], [], "gs")
        self.local, = self.ax.plot([], [], "k-", lw=2)
        self.ax.invert_yaxis()      # CARLA/UE y axis points right
        self.ax.set_aspect("equal")
        self.ax.set_title("Top down: orange = driven, black = planned")
        self.sp, = self.ax2.plot([], [], label="speed")
        self.ref, = self.ax2.plot([], [], "--", label="reference")
        self.ax2.legend()
        self.ax2.grid(True)
        self.ax2.set_xlabel("s")
        self.ax2.set_ylabel("m/s")

    def update(self, hist, lead, local_wp, t_ref):
        self.trace.set_data(hist["x"], hist["y"])
        self.car.set_data([hist["x"][-1]], [hist["y"][-1]])
        if lead is not None:
            self.lead.set_data([lead[0]], [lead[1]])
        if local_wp:
            arr = np.array(local_wp)
            self.local.set_data(arr[:, 0], arr[:, 1])
        self.sp.set_data(hist["t"], hist["v"])
        self.ref.set_data(hist["t"], hist["vref"])
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.plt.pause(0.001)


def save_outputs(hist, waypoints, fence, box_pts, collisions):
    import matplotlib
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wp = np.array(waypoints)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wp[:, 0], wp[:, 1], "g-", label="global waypoints")
    ax.plot(hist["x"], hist["y"], color=(1, .5, 0), label="ego")
    ax.plot([fence[0], fence[2]], [fence[1], fence[3]], "r-", lw=3, label="stop line")
    if box_pts:
        b = np.array(box_pts)
        ax.plot(b[:, 0], b[:, 1], "b+", label="parked car")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Vehicle trajectory")
    fig.savefig(os.path.join(OUTPUT_DIR, "trajectory.png"), dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(hist["t"], hist["v"], label="speed")
    axs[0].plot(hist["t"], hist["vref"], "--", label="reference")
    axs[0].legend(); axs[0].grid(True); axs[0].set_ylabel("m/s")
    axs[1].plot(hist["t"], hist["thr"], label="throttle")
    axs[1].plot(hist["t"], hist["brk"], label="brake")
    axs[1].legend(); axs[1].grid(True)
    axs[2].plot(hist["t"], hist["st"], label="steer")
    axs[2].legend(); axs[2].grid(True); axs[2].set_xlabel("s")
    fig.savefig(os.path.join(OUTPUT_DIR, "controls.png"), dpi=100)
    plt.close(fig)

    with open(os.path.join(OUTPUT_DIR, "trajectory.txt"), "w") as fh:
        for i in range(len(hist["t"])):
            fh.write("%3.3f, %3.3f, %2.3f, %6.3f\n" %
                     (hist["x"][i], hist["y"][i], hist["v"][i], hist["t"][i]))
    with open(os.path.join(OUTPUT_DIR, "collision_count.txt"), "w") as fh:
        fh.write(str(len(collisions)))


# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------
def run(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0 if args.record else 30.0)
    world = client.get_world()
    if args.list_cars:
        for b in sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda b: b.id):
            print(b.id)
        return
    if args.town and not world.get_map().name.endswith(args.town):
        print("Loading", args.town)
        world = client.load_world(args.town)
    world_map = world.get_map()
    print("CARLA server %s, map %s" % (client.get_server_version(), world_map.name))
    bp_lib = world.get_blueprint_library()

    # ---- scenario geometry from the map ------------------------------------
    if args.start_index is not None:
        spawn = world_map.get_spawn_points()[args.start_index]
        first = build_route(world_map.get_waypoint(spawn.location), 450.0,
                            WP_SPACING, args.turn)
        candidates = [(args.start_index, first)]
    else:
        candidates = find_starts(world_map, WP_SPACING, args.turn)
    candidates = [c for c in candidates if junction_span(c[1])[0] is not None]
    if not candidates:
        sys.exit("Could not find a lane that runs into an intersection on this "
                 "map. Try another map (--town Town01) or pass --start-index.")
    cruise = args.speed_kmh / 3.6
    existing = len(world.get_actors().filter("vehicle.*"))
    if existing:
        print("Note: %d vehicle(s) already in the world (e.g. the Editor's own car)." % existing)

    if args.dashboard and args.record:
        print("Note: --dashboard already records the video, ignoring --record.")
        args.record = False
    if args.headless and (args.dashboard or args.record):
        sys.exit("--headless turns rendering off, so cameras cannot be used. "
                 "Drop --headless, or drop --dashboard/--record.")
    if args.dashboard and args.video_size is None:
        args.video_size = "1280x720"
    if args.video_size is None:
        args.video_size = "1024x576"
    if args.weather and not hasattr(carla.WeatherParameters, args.weather):
        names = [n for n in dir(carla.WeatherParameters) if n[:1].isupper() and not n.startswith("_")]
        sys.exit("Unknown weather '%s'. Presets: %s" % (args.weather, ", ".join(names)))

    original_settings = world.get_settings()
    original_weather = None
    recorder_on = False
    actors = []
    collisions = []
    rig = None
    sink = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        if args.headless:
            settings.no_rendering_mode = True
        world.apply_settings(settings)
        if args.weather:
            original_weather = world.get_weather()
            world.set_weather(getattr(carla.WeatherParameters, args.weather))
            print("Weather:", args.weather)
        if args.save_replay:
            client.start_recorder(args.save_replay + ".log", True)
            recorder_on = True
            print("CARLA recorder started: %s.log" % args.save_replay)

        # ---- actors ---------------------------------------------------------
        # Try the candidate start points in order until one is free.
        ego_choices = EGO_BLUEPRINTS
        if args.ego_car:
            if len(bp_lib.filter(args.ego_car)) == 0:
                print("WARNING: no car matches '%s', using the default car. "
                      "List cars with --list-cars." % args.ego_car)
            else:
                ego_choices = [args.ego_car] + EGO_BLUEPRINTS
        ego = None
        for start_idx, cand_route in candidates[:20]:
            cand_route, waypoints, fence, j = make_waypoints(cand_route, cruise, WP_SPACING)
            ego = spawn_on_route(world, bp_lib, cand_route[0], ego_choices, "hero")
            if ego is not None:
                route = cand_route
                break
            print("Spawn point %s is blocked, trying the next one..." % start_idx)
        if ego is None:
            sys.exit("Could not spawn the ego vehicle at any candidate start point. "
                     "Stop the Editor's own car or restart Play and try again.")
        actors.append(ego)
        print("Start spawn point %s, intersection after %.0f m, %d waypoints"
              % (start_idx, j * WP_SPACING, len(waypoints)))

        parked, parked_pts = None, []
        if not args.no_parked:
            idx = max(8, min(int(0.3 * j), int(60.0 / WP_SPACING)))
            parked = spawn_parked_car(world, bp_lib, route, idx,
                                      ([args.parked_car] if args.parked_car else []) + PARKED_BLUEPRINTS)
            if parked is None:
                print("WARNING: could not spawn the parked car, continuing without it")
            else:
                actors.append(parked)

        lead, lead_ctrl, lead_wps = None, None, None
        if not args.no_lead:
            base = max(8, min(int(0.3 * j), int(60.0 / WP_SPACING))) + int(30.0 / WP_SPACING)
            base = min(base, j - 12)
            lead = spawn_on_route(world, bp_lib, route[base],
                                  ([args.lead_car] if args.lead_car else []) + LEAD_BLUEPRINTS, "lead")
            if lead is None:
                print("WARNING: could not spawn the lead car, continuing without it")
            else:
                actors.append(lead)
                # The lead keeps driving well past the end of the ego route,
                # otherwise it would sit there as a stopped car in front of us.
                lead_route = route[base:] + build_route(
                    route[-1], 150.0, WP_SPACING, args.turn)[1:]
                lead_wps = [[w.transform.location.x, w.transform.location.y,
                             args.lead_speed] for w in lead_route]
                lead_ctrl = control.PathController(max_steer_deg=max_steer_degrees(lead))
                lead_idx = 0

        col_sensor = world.spawn_actor(bp_lib.find("sensor.other.collision"),
                                       carla.Transform(), attach_to=ego)
        actors.append(col_sensor)
        col_sensor.listen(lambda ev: collisions.append(
            (ev.frame, ev.other_actor.type_id)))

        frames_dir = None

        # let everything settle on the road, held still
        for _ in range(20):
            send(ego, 0, 0, 1.0, hand_brake=True)
            if lead is not None:
                send(lead, 0, 0, 1.0, hand_brake=True)
            world.tick()
        if parked is not None:
            parked_pts = box_points(parked)

        draw_scene(world, waypoints, fence, parked_pts)

        dash = None
        need_rig = args.dashboard or args.aeb or args.radar_lead
        if need_rig:
            dash = None
            if args.dashboard:
                import dashboard as dashboard_mod
                width = int(args.video_size.lower().split("x")[0])
                dash = dashboard_mod.Dashboard(width, dt=FIXED_DELTA_SECONDS)
            rig = sensor_rig.SensorRig(
                world, bp_lib, ego,
                chase_size=dash.chase_size if dash else None,
                front_size=dash.front_size if dash else None,
                lidar=bool(dash) or args.aeb,
                radar=bool(dash) or args.radar_lead,
                dt=FIXED_DELTA_SECONDS)
            actors.extend(rig.sensors)
            if (args.aeb and not rig.has_lidar) or (args.radar_lead and not rig.has_radar):
                print("WARNING: a sensor needed for --aeb / --radar-lead is missing; "
                      "that feature will be inactive.")
            print("Sensors on the car: %s" % ", ".join(sorted(rig._queues) + ["lane invasion"]))
            print("Warming up the sensors, the first frames can be slow...")
            for _ in range(15):
                send(ego, 0, 0, 1.0, hand_brake=True)
                if lead is not None:
                    send(lead, 0, 0, 1.0, hand_brake=True)
                fr = world.tick()
                rig.collect(fr)
            if dash:
                sink = VideoSink(os.path.join(OUTPUT_DIR, "dashboard.mp4"),
                                 os.path.join(OUTPUT_DIR, "dashboard_frames"), args.record_fps)
                print("Dashboard video %dx%d at %d fps -> %s" %
                      (dash.W, dash.H, args.record_fps, sink.out_path))

        if args.record:
            width, height = [int(v) for v in args.video_size.lower().split("x")]
            frames_dir = os.path.join(OUTPUT_DIR, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for old in glob.glob(os.path.join(frames_dir, "*.png")):
                os.remove(old)
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(width))
            cam_bp.set_attribute("image_size_y", str(height))
            cam_bp.set_attribute("fov", "90")
            cam_bp.set_attribute("sensor_tick", str(1.0 / args.record_fps))
            cam = world.spawn_actor(cam_bp, carla.Transform(
                carla.Location(x=-8.5, z=3.2), carla.Rotation(pitch=-12.0)),
                attach_to=ego)
            actors.append(cam)
            counter = itertools.count()
            cam.listen(lambda image: image.save_to_disk(
                os.path.join(frames_dir, "%06d.png" % next(counter))))
            print("Recording %dx%d at %d fps to %s" %
                  (width, height, args.record_fps, frames_dir))

            print("Starting the recording camera, the first frames can be slow...")
            for _ in range(15):          # warm up the camera while the cars stand still
                send(ego, 0, 0, 1.0, hand_brake=True)
                if lead is not None:
                    send(lead, 0, 0, 1.0, hand_brake=True)
                world.tick()

        # ---- planner and controller ----------------------------------------
        obstacles = [box_polygon(parked)] if parked is not None else []
        planner = mp.MotionPlanner(waypoints, [fence], obstacles,
                                   ego.bounding_box.extent.x, ego.bounding_box.extent.y)
        controller = control.PathController(max_steer_deg=max_steer_degrees(ego))
        spectator = world.get_spectator()
        cam_mode = args.camera if args.follow_camera else "free"

        hist = dict(t=[], x=[], y=[], v=[], vref=[], thr=[], brk=[], st=[], vlead=[])
        half_len = ego.bounding_box.extent.x
        aeb = pc.AEBMonitor(half_len) if (args.aeb and rig is not None and rig.has_lidar) else None
        radar_track = pc.RadarLeadTracker() if (args.radar_lead and rig is not None and rig.has_radar) else None
        tracker = kpi.KPITracker(fence, half_len,
                                 lead.bounding_box.extent.x if lead is not None else 2.4,
                                 FIXED_DELTA_SECONDS)
        wp_arr = np.array(waypoints)
        path_len = 0.0
        last_xy = None
        dash_div = 20 // args.record_fps
        sens = {}
        live = LivePlot(waypoints, fence, parked_pts) if args.live_plot else None

        t0 = world.get_snapshot().timestamp.elapsed_seconds
        last_state = None
        local_wp = None
        reached = False
        frame = 0
        end_x, end_y = waypoints[-1][0], waypoints[-1][1]
        max_frames = int(args.max_time / FIXED_DELTA_SECONDS)
        print("Driving. Ctrl+C to stop.")

        while frame < max_frames:
            fr = world.tick()
            frame += 1
            if rig is not None:
                sens = rig.decode(rig.collect(fr))
            now = world.get_snapshot().timestamp.elapsed_seconds - t0
            x, y, yaw, speed = vehicle_state(ego)

            update_spectator(spectator, cam_mode, ego, x, y, yaw, frame)

            if now < WAIT_TIME_BEFORE_START:
                send(ego, 0, 0, 1.0)
                if lead is not None:
                    send(lead, 0, 0, 1.0)
                continue
            t = now - WAIT_TIME_BEFORE_START

            # ---- lead vehicle (scripted, no traffic manager needed) ----------
            lead_state = None
            lead_gt = None
            if lead is not None:
                lx, ly, lyaw, lspeed = vehicle_state(lead)
                lead_state = [lx, ly, lspeed]
                lead_gt = (lx, ly, lyaw, lspeed)
                if math.hypot(lx - lead_wps[-1][0], ly - lead_wps[-1][1]) < 3.0:
                    send(lead, 0, 0, 1.0, hand_brake=True)
                else:
                    # Nearest route waypoint (only ever moves forward), then hand
                    # the controller the next 40 m of the route.
                    while (lead_idx < len(lead_wps) - 2 and
                           math.hypot(lead_wps[lead_idx + 1][0] - lx, lead_wps[lead_idx + 1][1] - ly) <
                           math.hypot(lead_wps[lead_idx][0] - lx, lead_wps[lead_idx][1] - ly)):
                        lead_idx += 1
                    window = lead_wps[max(lead_idx - 1, 0):lead_idx + 20]
                    if len(window) >= 2:
                        lead_ctrl.set_path(window)
                    lt, ls, lb = lead_ctrl.step(lx, ly, lyaw, lspeed, FIXED_DELTA_SECONDS)
                    send(lead, lt, ls, lb)

            # ---- radar based lead detection (optional) --------------------------
            path_ego = None
            if local_wp:
                path_ego = pc.world_to_ego(np.array(local_wp)[:, 0:2], x, y, yaw)
            lead_source = "ground truth" if lead is not None else None
            if radar_track is not None:
                radar_track.update(sens.get("radar"), speed, path_ego)
                lead_state = radar_track.lead_state_world(x, y, yaw)
                lead_source = "radar"

            # ---- planning ----------------------------------------------------
            if frame % LP_FREQUENCY_DIVISOR == 0:
                wp = planner.update(x, y, yaw, speed, t, lead_state)
                if wp is not None:
                    local_wp = wp
                    controller.set_path(wp)
                draw_plan(world, planner)
                if planner.state != last_state:
                    print("t=%6.2f s  planner state: %s" %
                          (t, STATE_NAMES[planner.state]))
                    last_state = planner.state

            # ---- control -----------------------------------------------------
            if local_wp is not None:
                throttle, steer, brake = controller.step(x, y, yaw, speed, FIXED_DELTA_SECONDS)
            else:
                throttle, steer, brake = 0.0, 0.0, 1.0
            aeb_on = False
            if aeb is not None:
                aeb_on = aeb.update(sens.get("lidar"), path_ego, speed)
                if aeb_on:
                    throttle, brake = 0.0, 1.0
            send(ego, throttle, steer, brake)

            # ---- metrics (ground truth, independent of the planner) --------
            try:
                acc = ego.get_acceleration()
                acc_xy = (acc.x, acc.y)
            except Exception:
                acc_xy = None
            tracker.update(t, x, y, yaw, speed, acc_xy, lead_gt, STATE_NAMES[planner.state],
                           planner.state == HOLD_AT_LINE, aeb_on)
            if last_xy is not None:
                path_len += math.hypot(x - last_xy[0], y - last_xy[1])
            last_xy = (x, y)

            hist["t"].append(t); hist["x"].append(x); hist["y"].append(y)
            hist["v"].append(speed); hist["vref"].append(controller.desired_speed)
            hist["thr"].append(float(throttle)); hist["brk"].append(float(brake))
            hist["st"].append(float(steer))
            hist["vlead"].append(lead_gt[3] if lead_gt is not None else float("nan"))

            if sink is not None and frame % dash_div == 0:
                ego_pose = (x, y, yaw)
                ez = ego.get_transform().location.z
                boxes = []
                if lead is not None:
                    boxes.append(actor_box(lead, ego_pose, ez, (70, 220, 235),
                                           "lead %.0f m" % tracker.gap_now if tracker.gap_now is not None else "lead"))
                if parked is not None:
                    boxes.append(actor_box(parked, ego_pose, ez, (255, 176, 40), "parked car"))
                dbg = planner.debug
                cands = []
                for i, pth in enumerate(dbg.get("paths", [])):
                    xy = pc.world_to_ego(np.stack([pth[0], pth[1]], axis=1), x, y, yaw)
                    cands.append(dict(xy=xy, best=(i == dbg.get("best_index")),
                                      blocked=not bool(dbg["collision_check_array"][i])))
                near = np.hypot(wp_arr[:, 0] - x, wp_arr[:, 1] - y).argmin()
                route_ego = pc.world_to_ego(wp_arr[near:near + 25, 0:2], x, y, yaw)
                fence_ego = pc.world_to_ego(np.array([[fence[0], fence[1]], [fence[2], fence[3]]]), x, y, yaw)
                frame_data = dict(
                    chase=sens.get("chase"), front=sens.get("front"),
                    lidar=sens.get("lidar"), radar=sens.get("radar"),
                    path=path_ego, cands=cands, route=route_ego, fence=fence_ego,
                    boxes=boxes, hist=hist, ego_half_len=half_len,
                    ego_half_wid=ego.bounding_box.extent.y,
                    hud=dict(state=STATE_NAMES[planner.state], speed=speed,
                             vref=controller.desired_speed, t=t,
                             thr=float(throttle), brk=float(brake), steer=float(steer),
                             gap=tracker.gap_now, ttc=tracker.ttc_now, aeb=aeb_on,
                             collisions=len(collisions), lane=rig.lane_crossings,
                             lead_source=lead_source))
                sink.add(dash.render(frame_data))
            if live is not None and frame % 4 == 0:
                live.update(hist, lead_state, local_wp, t)

            if math.hypot(end_x - x, end_y - y) < 2.0:
                reached = True
                break

        send(ego, 0, 0, 1.0, hand_brake=True)
        for _ in range(10 if frames_dir else 1):   # let the last frames reach the disk
            world.tick()
        if frames_dir:
            time.sleep(1.0)
        print("Reached the end of the route." if reached
              else "Stopped before the end (time limit or Ctrl+C).")
        print("Collisions recorded: %d %s" % (len(collisions),
              sorted(set(c[1] for c in collisions)) if collisions else ""))
        if hist["t"]:
            save_outputs(hist, waypoints, fence, parked_pts, collisions)
            print("Plots and trajectory saved in", OUTPUT_DIR)
            rep_ = tracker.report(collisions, rig.lane_crossings if rig else 0, reached,
                                  hist["t"][-1], path_len, lead is not None)
            kpi.print_report(rep_)
            kpi.save_report(rep_, os.path.join(OUTPUT_DIR, "kpi_report.json"))
            print("Report saved in", os.path.join(OUTPUT_DIR, "kpi_report.json"))
        if sink is not None:
            sink.close()
            sink = None
        if frames_dir:
            encode_video(frames_dir, args.record_fps,
                         os.path.join(OUTPUT_DIR, "run.mp4"))
    finally:
        # Always leave the simulator running normally and remove our actors.
        # Settings go back to asynchronous mode first, so the destroy commands
        # below are not left waiting for a tick that never comes.
        if sink is not None:
            try:
                sink.close()
            except Exception:
                pass
        try:
            if recorder_on:
                client.stop_recorder()
        except Exception as exc:
            print("could not stop the recorder:", exc)
        try:
            world.apply_settings(original_settings)
        except Exception as exc:
            print("could not restore simulator settings:", exc)
        try:
            if original_weather is not None:
                world.set_weather(original_weather)
        except Exception as exc:
            print("could not restore the weather:", exc)
        try:
            for a in actors:
                if a.type_id.startswith("sensor."):
                    a.stop()
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        except Exception as exc:
            print("cleanup warning:", exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("-p", "--port", type=int, default=2000)
    ap.add_argument("--town", default=None,
                    help="load this map first (default: use the map already loaded)")
    ap.add_argument("--start-index", type=int, default=None,
                    help="spawn point index to start from (default: search automatically)")
    ap.add_argument("--turn", choices=["right", "left", "straight", "any"], default="right",
                    help="which way to go at forks in the road (default: right)")
    ap.add_argument("--speed-kmh", type=float, default=30.0, help="cruise speed (default 30)")
    ap.add_argument("--lead-speed", type=float, default=3.5, help="lead car speed in m/s")
    ap.add_argument("--no-lead", action="store_true")
    ap.add_argument("--no-parked", action="store_true")
    ap.add_argument("--live-plot", action="store_true", help="open a live matplotlib window")
    ap.add_argument("--camera", choices=["overhead", "chase", "free"], default="overhead",
                    help="Editor camera: overhead (default), chase (behind the car) or "
                         "free (the script never touches it, fly it yourself)")
    ap.add_argument("--no-follow-camera", dest="follow_camera", action="store_false",
                    help="same as --camera free")
    ap.add_argument("--ego-car", default=None,
                    help="car blueprint, e.g. vehicle.tesla.model3 or 'vehicle.audi.*'")
    ap.add_argument("--lead-car", default=None, help="car blueprint for the lead car")
    ap.add_argument("--parked-car", default=None, help="car blueprint for the parked car")
    ap.add_argument("--list-cars", action="store_true", help="print the car blueprints and exit")
    ap.add_argument("--record", action="store_true",
                    help="record the run from a camera on the car and make controller_output/run.mp4")
    ap.add_argument("--record-fps", type=int, default=20, choices=[10, 20],
                    help="video frame rate (20 = every simulation step)")
    ap.add_argument("--video-size", default=None,
                    help="video size, e.g. 1024x576 (default for --record) or 1280x720 "
                         "(default for --dashboard; 1920x1080 also works)")
    ap.add_argument("--dashboard", action="store_true",
                    help="record the multi-panel sensor dashboard video "
                         "(chase cam + HUD, front cam with lidar/radar, bird's eye view, charts)")
    ap.add_argument("--aeb", action="store_true",
                    help="enable the lidar based emergency brake as a safety layer")
    ap.add_argument("--radar-lead", action="store_true",
                    help="detect the lead car with the radar instead of using simulator ground truth")
    ap.add_argument("--weather", default=None,
                    help="weather preset, e.g. ClearNoon, WetCloudyNoon, HardRainNoon, ClearSunset")
    ap.add_argument("--headless", action="store_true",
                    help="no rendering (fast, GPU friendly validation run; no cameras, no dashboard)")
    ap.add_argument("--save-replay", default=None, metavar="NAME",
                    help="save a CARLA recorder file NAME.log you can replay and film from any angle")
    ap.add_argument("--max-time", type=float, default=150.0, help="time limit in sim seconds")
    args = ap.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nCancelled by user.")


if __name__ == "__main__":
    main()
