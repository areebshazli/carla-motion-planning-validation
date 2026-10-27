#!/usr/bin/env python3
"""Composite "dashboard" video frame: chase camera with HUD, front camera with
lidar / radar / 3D box overlays, a bird's eye view of lidar + radar + the
planner's paths, and rolling charts. Pure NumPy + Pillow, no carla import.

Layout on the 1280x720 base canvas (everything scales with --video-size):

    +-------------------------------+------------------+
    |  chase camera + HUD (800x450) | front cam (480x270)
    |                               +------------------+
    |                               | bird's eye       |
    +-------------------------------+ lidar+radar+plan |
    |  rolling charts (800x270)     | (480x450)        |
    +-------------------------------+------------------+
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import perception as pc

BASE_W, BASE_H = 1280, 720
CHASE = (0, 0, 800, 450)
FRONT = (800, 0, 480, 270)
BEV = (800, 270, 480, 450)
CHARTS = (0, 450, 800, 270)

BG = (14, 16, 20)
PANEL_BG = (20, 23, 29)
GRID = (52, 58, 70)
TEXT = (232, 236, 242)
MUTED = (150, 158, 172)
GREEN = (80, 220, 120)
AMBER = (255, 176, 40)
RED = (255, 80, 70)
BLUE = (90, 150, 255)
CYAN = (70, 220, 235)
ORANGE = (255, 140, 0)

STATE_COLOURS = {"CRUISE": GREEN, "APPROACH_STOP": AMBER, "HOLD_AT_LINE": RED}

_VIR = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)


def cmap(t):
    """0..1 -> viridis-like uint8 colours, shape (N, 3)."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0) * (len(_VIR) - 1)
    i = np.minimum(t.astype(int), len(_VIR) - 2)
    f = (t - i)[:, None]
    return (_VIR[i] * (1 - f) + _VIR[i + 1] * f).astype(np.uint8)


def _even(v):
    v = int(round(v))
    return v + (v % 2)


class Dashboard(object):
    def __init__(self, width=1280, window_s=12.0, dt=0.05, fov_front=90.0):
        self.s = float(width) / BASE_W
        self.W = _even(BASE_W * self.s)
        self.H = _even(BASE_H * self.s)
        self.window_s = window_s
        self.dt = dt
        self.fov_front = fov_front
        self._fonts = {}

    # sizes the cameras should be created with
    @property
    def chase_size(self):
        return self._px(CHASE[2]), self._px(CHASE[3])

    @property
    def front_size(self):
        return self._px(FRONT[2]), self._px(FRONT[3])

    def _px(self, v):
        return int(round(v * self.s))

    def _font(self, size):
        size = max(8, int(round(size * self.s)))
        if size not in self._fonts:
            font = None
            for name in ("DejaVuSans.ttf", "segoeui.ttf", "arial.ttf", "Arial.ttf",
                         "LiberationSans-Regular.ttf"):
                try:
                    font = ImageFont.truetype(name, size)
                    break
                except Exception:
                    continue
            if font is None:
                try:
                    font = ImageFont.load_default(size)
                except Exception:
                    font = ImageFont.load_default()
            self._fonts[size] = font
        return self._fonts[size]

    def _text(self, draw, xy, text, size=14, fill=TEXT, anchor="la", shadow=True):
        font = self._font(size)
        x, y = xy
        if shadow:
            draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0), anchor=anchor)
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

    # ------------------------------------------------------------------
    def render(self, f):
        canvas = Image.new("RGB", (self.W, self.H), BG)
        canvas.paste(self._chase_panel(f), (self._px(CHASE[0]), self._px(CHASE[1])))
        canvas.paste(self._front_panel(f), (self._px(FRONT[0]), self._px(FRONT[1])))
        canvas.paste(self._bev_panel(f), (self._px(BEV[0]), self._px(BEV[1])))
        canvas.paste(self._chart_panel(f), (self._px(CHARTS[0]), self._px(CHARTS[1])))
        return np.asarray(canvas)

    # ------------------------------------------------------------------
    def _camera_image(self, arr, size):
        if arr is None:
            img = Image.new("RGB", size, PANEL_BG)
            ImageDraw.Draw(img).text((size[0] // 2, size[1] // 2), "no camera image",
                                     font=self._font(16), fill=MUTED, anchor="mm")
            return img
        img = Image.fromarray(arr)
        if img.size != size:
            img = img.resize(size, Image.BILINEAR)
        return img

    # ---- chase camera + HUD ------------------------------------------
    def _chase_panel(self, f):
        size = (self._px(CHASE[2]), self._px(CHASE[3]))
        img = self._camera_image(f.get("chase"), size).convert("RGBA")
        hud = f.get("hud", {})
        ov = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        s = self.s
        # info card
        card = (int(12 * s), int(12 * s), int(300 * s), int(150 * s))
        d.rounded_rectangle(card, radius=int(10 * s), fill=(8, 10, 14, 170))
        state = hud.get("state", "")
        col = STATE_COLOURS.get(state, TEXT)
        self._text(d, (card[0] + 14 * s, card[1] + 10 * s), state.replace("_", " "), 15, col)
        speed = hud.get("speed", 0.0)
        self._text(d, (card[0] + 14 * s, card[1] + 32 * s), "%.0f" % (speed * 3.6), 40, TEXT, shadow=False)
        self._text(d, (card[0] + 92 * s, card[1] + 50 * s), "km/h", 14, MUTED, shadow=False)
        self._text(d, (card[0] + 14 * s, card[1] + 82 * s),
                   "%.1f m/s   target %.1f" % (speed, hud.get("vref", 0.0)), 13, MUTED, shadow=False)
        gap, ttc = hud.get("gap"), hud.get("ttc")
        self._text(d, (card[0] + 14 * s, card[1] + 102 * s),
                   "gap to lead  %s" % ("%.1f m" % gap if gap is not None else "--"), 13, TEXT, shadow=False)
        self._text(d, (card[0] + 14 * s, card[1] + 120 * s),
                   "time to collision  %s" % ("%.1f s" % ttc if ttc is not None else "--"), 13, TEXT, shadow=False)
        # clock and counters, top right
        self._text(d, (size[0] - 14 * s, 14 * s), "t = %.1f s" % hud.get("t", 0.0), 16, TEXT, "ra")
        coll = hud.get("collisions", 0)
        self._text(d, (size[0] - 14 * s, 36 * s),
                   "collisions %d" % coll, 13, RED if coll else GREEN, "ra")
        self._text(d, (size[0] - 14 * s, 54 * s),
                   "lane crossings %d" % hud.get("lane", 0), 13, MUTED, "ra")
        src = hud.get("lead_source")
        if src:
            self._text(d, (size[0] - 14 * s, 72 * s), "lead from " + src, 13, MUTED, "ra")
        if hud.get("aeb"):
            bw, bh = int(250 * s), int(44 * s)
            x0 = (size[0] - bw) // 2
            d.rounded_rectangle((x0, int(14 * s), x0 + bw, int(14 * s) + bh), radius=int(8 * s),
                                fill=(220, 30, 30, 215))
            self._text(d, (size[0] // 2, int(14 * s) + bh // 2), "EMERGENCY BRAKE", 20, (255, 255, 255), "mm", False)
        # pedal and steering bars along the bottom
        by = size[1] - int(34 * s)
        bw = int(150 * s)
        bh = int(12 * s)

        def bar(x, label, value, colour, centred=False):
            d.rounded_rectangle((x, by, x + bw, by + bh), radius=int(5 * s), fill=(8, 10, 14, 170))
            if centred:
                mid = x + bw // 2
                x0, x1 = sorted((mid, mid + int(value * bw / 2)))
                d.rectangle((x0, by, max(x1, x0 + 1), by + bh), fill=colour + (230,))
                d.line((mid, by - 2, mid, by + bh + 2), fill=(255, 255, 255, 180), width=max(1, int(s)))
            else:
                d.rounded_rectangle((x, by, x + max(int(np.clip(value, 0, 1) * bw), 2), by + bh),
                                    radius=int(5 * s), fill=colour + (230,))
            self._text(d, (x, by - 16 * s), label, 12, MUTED, shadow=False)

        bar(int(14 * s), "throttle", hud.get("thr", 0.0), GREEN)
        bar(int(14 * s) + bw + int(14 * s), "brake", hud.get("brk", 0.0), RED)
        bar(int(14 * s) + 2 * (bw + int(14 * s)), "steer", hud.get("steer", 0.0), BLUE, True)
        return Image.alpha_composite(img, ov).convert("RGB")

    # ---- front camera ---------------------------------------------------
    def _front_panel(self, f):
        size = (self._px(FRONT[2]), self._px(FRONT[3]))
        img = self._camera_image(f.get("front"), size)
        w, h = size
        lidar, radar = f.get("lidar"), f.get("radar")
        if lidar is not None and len(lidar):
            u, v, depth, vis = pc.project_ego_points(lidar[:, 0:3], w, h, self.fov_front)
            m = vis & (lidar[:, 2] > 0.05) & (depth < 60.0)
            if m.any():
                arr = np.array(img)
                col = cmap(1.0 - (depth[m] - 3.0) / 37.0)
                uu, vv = u[m].astype(int), v[m].astype(int)
                k = max(1, int(round(1.5 * self.s)))
                for du in range(k):
                    for dv in range(k):
                        arr[np.clip(vv + dv, 0, h - 1), np.clip(uu + du, 0, w - 1)] = col
                img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        for b in f.get("boxes", []):
            verts = b.get("verts")
            if verts is None:
                continue
            u, v, depth, vis = pc.project_ego_points(verts, w, h, self.fov_front)
            front_ok = depth > 0.5
            for a, c in pc.BOX_EDGES:
                if front_ok[a] and front_ok[c]:
                    d.line((u[a], v[a], u[c], v[c]), fill=b.get("colour", CYAN), width=max(1, int(2 * self.s)))
            if front_ok.all():
                top = int(np.argmin(v))
                lbl = b.get("label", "")
                if lbl:
                    self._text(d, (float(np.clip(np.mean(u), 4, w - 4)), float(max(v.min() - 4, 12 * self.s))),
                               lbl, 12, b.get("colour", CYAN), "ms")
        if radar is not None and len(radar):
            u, v, depth, vis = pc.project_ego_points(radar[:, 0:3], w, h, self.fov_front)
            r = max(3, int(5 * self.s))
            for i in np.nonzero(vis)[0]:
                vel = radar[i, 3]
                c = RED if vel < -1.0 else (BLUE if vel > 1.0 else (240, 240, 240))
                d.ellipse((u[i] - r, v[i] - r, u[i] + r, v[i] + r), outline=c, width=max(1, int(2 * self.s)))
        self._text(d, (8 * self.s, 6 * self.s), "FRONT CAMERA + LIDAR (colour = range) + RADAR", 11, TEXT)
        return img

    # ---- bird's eye view ----------------------------------------------------
    def _bev_panel(self, f):
        s = self.s
        bw, bh = self._px(BEV[2]), self._px(BEV[3])
        sc = 10.0 * s                       # pixels per metre
        cx, cy = bw / 2.0, 340.0 * s        # ego position on the panel

        def to_px(pts):
            pts = np.asarray(pts, dtype=float).reshape((-1, 2))
            return np.stack([cx + pts[:, 1] * sc, cy - pts[:, 0] * sc], axis=1)

        arr = np.zeros((bh, bw, 3), np.uint8)
        arr[:] = (12, 14, 18)
        lidar = f.get("lidar")
        if lidar is not None and len(lidar):
            uv = to_px(lidar[:, 0:2])
            u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
            inside = (u >= 0) & (u < bw - 1) & (v >= 0) & (v < bh - 1)
            u, v, z = u[inside], v[inside], lidar[inside, 2]
            k = max(1, int(round(1.5 * s)))
            ground = z < 0.3
            order = np.concatenate([np.nonzero(ground)[0], np.nonzero(~ground)[0]])
            col = np.empty((len(z), 3), np.uint8)
            col[ground] = (58, 68, 84)
            col[~ground] = cmap((z[~ground] - 0.3) / 2.0)
            for du in range(k):
                for dv in range(k):
                    arr[v[order] + dv, u[order] + du] = col[order]
        img = Image.fromarray(arr)
        ov = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # range rings
        for r in (10, 20, 30):
            rp = r * sc
            d.arc((cx - rp, cy - rp, cx + rp, cy + rp), 180 + 25, 360 - 25, fill=GRID + (255,), width=1)
            self._text(d, (cx + 4, cy - rp - 14 * s), "%d m" % r, 10, MUTED, shadow=False)
        d.line((cx, 0, cx, bh), fill=GRID + (120,), width=1)

        path, cands = f.get("path"), f.get("cands", [])
        aeb = f.get("hud", {}).get("aeb")
        if aeb and path is not None and len(path) > 1:
            d.line([tuple(p) for p in to_px(path)], fill=(255, 40, 40, 90),
                   width=int(2 * pc.AEB_HALF_WIDTH * sc), joint="curve")
        for c in cands:
            if len(c["xy"]) > 1:
                if c.get("best"):
                    continue
                d.line([tuple(p) for p in to_px(c["xy"])],
                       fill=((200, 80, 80, 160) if c.get("blocked") else (90, 150, 255, 150)),
                       width=max(1, int(s)))
        route = f.get("route")
        if route is not None and len(route):
            for p in to_px(route):
                d.ellipse((p[0] - 2 * s, p[1] - 2 * s, p[0] + 2 * s, p[1] + 2 * s), fill=GREEN + (200,))
        fence = f.get("fence")
        if fence is not None:
            d.line([tuple(p) for p in to_px(fence)], fill=RED + (255,), width=max(2, int(3 * s)))
        if path is not None and len(path) > 1:
            d.line([tuple(p) for p in to_px(path)], fill=ORANGE + (255,), width=max(2, int(3 * s)), joint="curve")
        for b in f.get("boxes", []):
            poly = b.get("poly")
            if poly is not None:
                pts = [tuple(p) for p in to_px(poly)]
                d.polygon(pts, outline=b.get("colour", CYAN) + (255,), fill=b.get("colour", CYAN) + (60,))
        # ego car
        half_l = f.get("ego_half_len", 2.4)
        half_w = f.get("ego_half_wid", 0.95)
        ego = pc.box_corners(0, 0, 0, half_l, half_w)
        d.polygon([tuple(p) for p in to_px(ego)], fill=(240, 240, 245, 255), outline=(30, 30, 30, 255))
        nose = to_px([[half_l + 1.2, 0], [half_l - 0.2, 0.7], [half_l - 0.2, -0.7]])
        d.polygon([tuple(p) for p in nose], fill=ORANGE + (255,))
        # radar detections
        radar = f.get("radar")
        if radar is not None and len(radar):
            r = max(3, 4 * s)
            for p, vel in zip(to_px(radar[:, 0:2]), radar[:, 3]):
                c = RED if vel < -1.0 else (BLUE if vel > 1.0 else (240, 240, 240))
                d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), outline=c + (255,), width=max(1, int(2 * s)))
        # legend
        ly = bh - 18 * s
        items = [("lidar", (94, 201, 98)), ("radar", RED), ("chosen path", ORANGE), ("alternatives", BLUE), ("stop line", RED)]
        x = 8 * s
        font = self._font(10)
        for name, col in items:
            d.rectangle((x, ly + 3 * s, x + 8 * s, ly + 11 * s), fill=col + (255,))
            self._text(d, (x + 11 * s, ly), name, 10, MUTED, shadow=False)
            x += 11 * s + d.textlength(name, font=font) + 12 * s
        self._text(d, (8 * s, 6 * s), "BIRD'S EYE: LIDAR + RADAR + PLANNED PATHS", 11, TEXT)
        return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

    # ---- charts -------------------------------------------------------------
    def _chart_panel(self, f):
        s = self.s
        w, h = self._px(CHARTS[2]), self._px(CHARTS[3])
        img = Image.new("RGB", (w, h), PANEL_BG)
        d = ImageDraw.Draw(img)
        hist = f.get("hist") or {}
        t_now = hud_t = f.get("hud", {}).get("t", 0.0)
        n = int(self.window_s / self.dt) + 1
        t = np.asarray(hist.get("t", [])[-n:], dtype=float)
        pad = 14 * s
        half = w / 2.0
        vmax = 6.0
        for key in ("v", "vref", "vlead"):
            vals = [x for x in hist.get(key, [])[-n:] if x is not None and not (isinstance(x, float) and math.isnan(x))]
            if vals:
                vmax = max(vmax, max(vals) * 1.15)
        self._plot(d, (pad, 34 * s, half - pad, h - 30 * s), t, t_now,
                   [("speed", hist.get("v", [])[-n:], CYAN),
                    ("target", hist.get("vref", [])[-n:], AMBER),
                    ("lead car", hist.get("vlead", [])[-n:], (190, 130, 255))],
                   (0.0, vmax), "SPEED (m/s)")
        self._plot(d, (half + pad, 34 * s, w - pad, h - 30 * s), t, t_now,
                   [("throttle", hist.get("thr", [])[-n:], GREEN),
                    ("brake", hist.get("brk", [])[-n:], RED),
                    ("steer", hist.get("st", [])[-n:], BLUE)],
                   (-1.05, 1.05), "CONTROLS")
        return img

    def _plot(self, d, rect, t, t_now, series, ylim, title):
        s = self.s
        x0, y0, x1, y1 = rect
        self._text(d, (x0, y0 - 26 * s), title, 12, TEXT, shadow=False)
        d.rectangle(rect, outline=GRID, width=1)
        lo, hi = ylim
        for k in range(1, 4):
            yy = y0 + (y1 - y0) * k / 4.0
            d.line((x0, yy, x1, yy), fill=(34, 39, 48), width=1)
            val = hi - (hi - lo) * k / 4.0
            self._text(d, (x0 + 3 * s, yy - 12 * s), "%.1f" % val, 9, MUTED, shadow=False)
        t_lo = t_now - self.window_s
        xs_lbl = x1 - 2
        lx = x0 + 70 * s
        for name, vals, col in series:
            self._text(d, (lx, y1 + 6 * s), name, 11, col, shadow=False)
            lx += 80 * s
            if len(vals) < 2 or len(t) < 2:
                continue
            m = min(len(vals), len(t))
            pts = []
            for ti, vi in zip(t[-m:], vals[-m:]):
                if vi is None or (isinstance(vi, float) and math.isnan(vi)):
                    if len(pts) > 1:
                        d.line(pts, fill=col, width=max(1, int(2 * s)))
                    pts = []
                    continue
                px = x0 + (x1 - x0) * (ti - t_lo) / self.window_s
                py = y1 - (y1 - y0) * (float(vi) - lo) / (hi - lo)
                pts.append((float(np.clip(px, x0, x1)), float(np.clip(py, y0, y1))))
            if len(pts) > 1:
                d.line(pts, fill=col, width=max(1, int(2 * s)))
        self._text(d, (x1, y1 + 6 * s), "last %d s" % self.window_s, 10, MUTED, "ra", shadow=False)
