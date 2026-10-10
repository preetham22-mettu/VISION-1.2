import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MP = True
except Exception:  # mediapipe not installed
    HAS_MP = False

# Resolution:  --sd (1280x720) | default (1280x800) | --hd (1920x1080)
if "--sd" in sys.argv:
    W, H = 1280, 720
elif "--hd" in sys.argv:
    W, H = 1920, 1080
else:
    W, H = 1280, 800
K = H / 720.0                    # scale factor for sizes/thickness
GLOW = [1.0]                     # glow strength (change live with [ and ])
WIN = "VISION"
F = 1.5 * H                      # focal length
AUTO_SPEED = 0.25                # rad/sec (slow continuous rotation)
ZOOM_MIN, ZOOM_MAX = 0.45, 3.6

# ---- BLUE palette (BGR) ----
C_BRIGHT = (255, 225, 110)
C_MID = (255, 160, 45)
C_DEEP = (215, 95, 20)
C_TEXT = (255, 235, 140)
C_WHITE = (255, 250, 235)

# ---- While Vision speaks the interface turns red (vision.py writes this file) ----
STATE_FILE = os.path.join(tempfile.gettempdir(), "vision_state.txt")
_BLUE = dict(C_BRIGHT=C_BRIGHT, C_MID=C_MID, C_DEEP=C_DEEP, C_TEXT=C_TEXT, C_WHITE=C_WHITE)
_RED = dict(C_BRIGHT=(70, 70, 255), C_MID=(38, 42, 232), C_DEEP=(18, 22, 190),
            C_TEXT=(135, 145, 255), C_WHITE=(235, 240, 255))
RED_K = [0.0]                    # 0 = blue, 1 = red (smooth transition)
_state_cache = [0.0, False]


def vision_speaking():
    """Read the state file (every ~80 ms). True if it says 'speaking' and is fresh."""
    now = time.time()
    if now - _state_cache[0] > 0.08:
        _state_cache[0] = now
        try:
            with open(STATE_FILE, "r") as f:
                txt = f.read().strip()
            fresh = (now - os.path.getmtime(STATE_FILE)) < 120      # avoid getting stuck on red after a crash
            _state_cache[1] = (txt == "speaking" and fresh)
        except Exception:
            _state_cache[1] = False
    return _state_cache[1]


def set_palette(k):
    """Blend blue <-> red. Drawing functions read these globals at call time."""
    g = globals()
    for name in _BLUE:
        g[name] = tuple(int(_BLUE[name][i] * (1 - k) + _RED[name][i] * k) for i in range(3))


def col(c, b=1.0):
    return tuple(int(max(0, min(255, v * b))) for v in c)


def mix(c, w, a):
    """Mix colour c towards w (white) by amount a."""
    a = max(0.0, min(1.0, a))
    return tuple(int(c[i] * (1 - a) + w[i] * a) for i in range(3))


random.seed(7)
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ geometry
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def icosphere(sub=1):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
         (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    cache = {}

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k in cache:
            return cache[k]
        p = verts[a] + verts[b]
        verts.append(p / np.linalg.norm(p))
        cache[k] = len(verts) - 1
        return cache[k]

    for _ in range(sub):
        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    edges = set()
    for a, b, c in f:
        for p, q in ((a, b), (b, c), (c, a)):
            edges.add((min(p, q), max(p, q)))
    return np.array(verts), np.array(sorted(edges))


CORE_V, CORE_E = icosphere(1)
INNER_V, INNER_E = icosphere(0)

# Small faceted crystal at the centre.
CRYSTAL_V = np.array([
    (0, 1, 0), (1, 0, 0), (0, 0, 1),
    (-1, 0, 0), (0, 0, -1), (0, -1, 0)
], dtype=float)
CRYSTAL_F = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
             (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)]
CRYSTAL_E = sorted({tuple(sorted(edge)) for face in CRYSTAL_F
                    for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))})

# latitude rings (dense near the equator)
u = np.linspace(-1, 1, 24)
LAT_ANG = np.sign(u) * (np.abs(u) ** 1.6) * 1.35
_a = np.linspace(0, 2 * math.pi, 72)
LAT_RINGS = [np.stack([math.cos(g) * np.cos(_a), np.full_like(_a, math.sin(g)),
                       math.cos(g) * np.sin(_a)], 1) for g in LAT_ANG]

# fine circle (more points for smooth outer rings)
_b = np.linspace(0, 2 * math.pi, 181)
UNIT_CIRCLE = np.stack([np.cos(_b), np.zeros_like(_b), np.sin(_b)], 1)

# Gyroscope rings: (radius, tilt_x, tilt_z, spin speed, comet speed, comet tail)
GYROS = [
    (1.00, 1.15, 0.00, 0.30, 0.22, 0.20),
    (1.00, 0.35, 1.05, -0.24, -0.17, 0.20),
    (1.00, -0.75, -0.80, 0.18, 0.27, 0.16),
    (1.24, 0.55, 0.35, -0.10, 0.12, 0.10),
]

# tilted orbits
ORBITS = []
for _ in range(8):
    a = rng.uniform(0.3, 0.75)
    b = a * rng.uniform(0.45, 0.85)
    R_ = rot_x(rng.uniform(0, 3.14)) @ rot_y(rng.uniform(0, 3.14)) @ rot_z(rng.uniform(0, 3.14))
    pts = np.stack([a * np.cos(_a), b * np.sin(_a), np.zeros_like(_a)], 1)
    ORBITS.append((pts @ R_.T, rng.uniform(0.08, 0.22) * rng.choice([-1, 1])))


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


SHELL = fib_sphere(340)


def rand_dirs(n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


NODES = rand_dirs(18) * rng.uniform(1.9, 3.25, (18, 1))
STARS = rand_dirs(48) * rng.uniform(2.1, 4.1, (48, 1))

TOKENS = ["0xFF3A", "TLS 1.3", "AES-256", "RSA 4096", "SYNC OK", "200 OK", "ACK", ">>> RDY",
          "sys.init()", "malloc()", "fork()", "exec()", "spawn()", "void*", "fn main", "impl Orb",
          "arc::new", "async ()", "use std", "ptr_ref", "mutex.lk", "kernel.d", "core.0", "hash256",
          "SIGTERM", "01101001", "10110100", "IRQ 0x7", "DMA xfer", "TCP/SYN", "HTTP/2", "eth0: UP",
          "pipe |>", "::bind", ".unwrap", "chmod +x", "HEAP 4K", "REG EAX", "PATCH /", ">> SCAN", "FAULT 0",
          "latency"]
LAB_DIR = rand_dirs(140)
LAB_DIR[:, 1] *= 0.85
LAB_POS = LAB_DIR * (0.35 + 0.72 * rng.random((140, 1)) ** 0.5)
LAB_TXT = [random.choice(TOKENS) for _ in range(140)]


# ------------------------------------------------------------------ rendering
def project(P, R, camd):
    c = P @ R.T
    depth = np.maximum(camd + c[:, 2], 0.12)
    s = F / depth
    x = np.clip(W / 2 + c[:, 0] * s, -30000, 30000)
    y = np.clip(H / 2 - c[:, 1] * s, -30000, 30000)
    return x, y, s, c[:, 2]


def poly(layer, P, R, camd, color, th=1):
    x, y, _, _ = project(P, R, camd)
    pts = np.stack([x, y], 1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(layer, [pts], False, color, th, cv2.LINE_AA)


def shaded_ring(layer, glow, P, R, camd, color, th, gth, bmin, bmax, t=0.0, speed=0.0, tail=0.15):
    """Thin depth-shaded ring: front side bright, back side dim.
    If speed != 0 a glowing comet (white-hot head + tail) travels around the ring.
    Core line goes on `layer` (sharp), halo goes on `glow` (for bloom)."""
    x, y, s, z = project(P, R, camd)
    n = len(P) - 1
    xs = x.astype(np.int32).tolist()
    ys = y.astype(np.int32).tolist()
    zm = (z[:-1] + z[1:]) * 0.5
    front = np.clip(0.5 - zm / 2.0, 0.0, 1.0)
    br = bmin + (bmax - bmin) * front
    boost = np.zeros(n)
    head = 0
    if speed:
        head_f = ((t * speed) % 1.0) * n
        d = (head_f - np.arange(n)) % n
        boost = np.clip(1.0 - d / max(1.0, tail * n), 0.0, 1.0) ** 1.5
        head = int(head_f) % n
    brl, bol = br.tolist(), boost.tolist()
    for i in range(n):
        bo = bol[i]
        c = col(mix(color, C_WHITE, 0.35 + bo * 0.6), brl[i] + bo * 0.9)
        p1, p2 = (xs[i], ys[i]), (xs[i + 1], ys[i + 1])
        cv2.line(layer, p1, p2, c, th + (1 if bo > 0.55 else 0), cv2.LINE_AA)
        if gth:
            cv2.line(glow, p1, p2, col(color, brl[i] * 0.85 + bo * 1.1), gth, cv2.LINE_AA)
    if speed:
        hp, hb = (xs[head], ys[head]), brl[head]
        r1 = max(2, int(2.2 * K))
        cv2.circle(glow, hp, r1 * 4, col(C_BRIGHT, 0.5 + hb * 0.8), -1, cv2.LINE_AA)
        cv2.circle(layer, hp, r1, col(C_WHITE, 0.6 + hb), -1, cv2.LINE_AA)


def tick_ring(layer, glow, R, camd, radius, n, t, tilt, spin, color, bmin, bmax, yoff=0.0):
    """Clock / scale style ring with tick marks."""
    k = np.arange(n)
    ang = k * 2 * math.pi / n + spin * t
    ln = np.where(k % 10 == 0, 0.075, np.where(k % 5 == 0, 0.045, 0.022))
    ca, sa = np.cos(ang), np.sin(ang)
    inner = np.stack([radius * ca, np.zeros(n), radius * sa], 1) @ tilt.T
    outer = np.stack([(radius + ln) * ca, np.zeros(n), (radius + ln) * sa], 1) @ tilt.T
    inner[:, 1] += yoff
    outer[:, 1] += yoff
    x1, y1, _, z1 = project(inner, R, camd)
    x2, y2, _, _ = project(outer, R, camd)
    front = np.clip(0.5 - z1 / 2.0, 0.0, 1.0)
    for i in range(n):
        longt = (i % 10 == 0)
        b = bmin + (bmax - bmin) * front[i] * (1.0 if longt else 0.7)
        p1, p2 = (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i]))
        cv2.line(layer, p1, p2, col(mix(color, C_WHITE, 0.3), b), 1, cv2.LINE_AA)
        if longt:
            cv2.line(glow, p1, p2, col(color, b * 0.9), max(2, int(2 * K)), cv2.LINE_AA)


def draw_hud(layer, glow, center, rad, t, zoom):
    """Screen-space rotating HUD arcs + tick marks around the sphere."""
    if rad < 30 or rad > 4 * W:
        return
    th2 = max(2, int(2 * K))
    r1 = int(rad * 1.40)
    for i in range(12):
        a0 = i * 30 + t * 10
        cv2.ellipse(layer, center, (r1, r1), 0, a0, a0 + 20, col(C_MID, 0.75), 1, cv2.LINE_AA)
    r2 = int(rad * 1.48)
    for i in range(3):
        a0 = i * 120 - t * 16
        cv2.ellipse(layer, center, (r2, r2), 0, a0, a0 + 62, col(C_WHITE, 0.95), 1, cv2.LINE_AA)
        cv2.ellipse(glow, center, (r2, r2), 0, a0, a0 + 62, col(C_BRIGHT, 1.0), th2 * 2, cv2.LINE_AA)
    r3 = rad * 1.56
    for i in range(72):
        a = math.radians(i * 5 + t * 4)
        ln = (9 if i % 6 == 0 else 4) * K
        ca, sa = math.cos(a), math.sin(a)
        p1 = (int(center[0] + ca * r3), int(center[1] + sa * r3))
        p2 = (int(center[0] + ca * (r3 + ln)), int(center[1] + sa * (r3 + ln)))
        cv2.line(layer, p1, p2, col(C_BRIGHT, 0.55 if i % 6 else 0.9), 1, cv2.LINE_AA)


def draw_title(layer, glow, t):
    """'VISION' title at the top centre: neon text with glow."""
    text = "V I S I O N"
    fs = 1.05 * K
    th = max(1, int(2 * K))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x0 = int(W / 2 - tw / 2)
    y0 = int(28 * K + tht + 14 * K)
    pulse = 0.88 + 0.12 * math.sin(t * 2.0)
    cv2.putText(glow, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fs,
                col(C_BRIGHT, 1.0 * pulse), th + max(3, int(3 * K)), cv2.LINE_AA)
    cv2.putText(layer, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fs,
                col(mix(C_BRIGHT, C_WHITE, 0.55), pulse), th, cv2.LINE_AA)
    # thin side lines
    ly = y0 - tht // 2
    gap, ln = int(26 * K), int(150 * K)
    for sgn in (-1, 1):
        xa = int(W / 2 + sgn * (tw / 2 + gap))
        xb = xa + sgn * ln
        cv2.line(layer, (xa, ly), (xb, ly), col(C_BRIGHT, 0.8 * pulse), 1, cv2.LINE_AA)
        cv2.line(glow, (xa, ly), (xb, ly), col(C_BRIGHT, 0.8 * pulse), max(3, int(3 * K)), cv2.LINE_AA)
        cv2.circle(layer, (xa, ly), max(2, int(2.5 * K)), col(C_WHITE, 1.0), -1, cv2.LINE_AA)


_BG = {}


def get_bg(k=0.0):
    key = (W, H)
    if key not in _BG:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        g = np.clip(1.0 - d * 0.85, 0, 1) ** 2
        blue = np.zeros((H, W, 3), np.uint8)
        blue[..., 0] = (g * 36 + 3).astype(np.uint8)
        blue[..., 1] = (g * 13 + 1).astype(np.uint8)
        blue[..., 2] = (g * 2).astype(np.uint8)
        red = np.zeros((H, W, 3), np.uint8)
        red[..., 0] = (g * 2).astype(np.uint8)
        red[..., 1] = (g * 5 + 1).astype(np.uint8)
        red[..., 2] = (g * 34 + 3).astype(np.uint8)
        _BG.clear()
        _BG[key] = (blue, red)
    blue, red = _BG[key]
    if k < 0.01:
        return blue
    if k > 0.99:
        return red
    return cv2.addWeighted(blue, 1.0 - k, red, k, 0)


def render(yaw, pitch, zoom, t):
    R = rot_x(pitch) @ rot_y(yaw)
    camd = 4.0 / zoom
    base = F / 4.0
    T1 = max(1, round(0.6 * K))      # fine single line
    T2 = max(2, round(2 * K))        # thick line
    GTH = max(4, round(5 * K))       # halo line thickness (glow layer)
    layer = np.zeros((H, W, 3), np.uint8)
    glow_layer = np.zeros_like(layer)

    # Twinkling stars
    x, y, s, cz = project(STARS, R, camd)
    for i, (xi, yi, si) in enumerate(zip(x, y, s)):
        if 0 <= xi < W and 0 <= yi < H:
            tw = 0.55 + 0.45 * math.sin(t * 2.3 + i * 1.7)
            radius = max(1, int(1.0 * K * si / base))
            px, py = int(xi), int(yi)
            cv2.circle(layer, (px, py), radius, col(C_BRIGHT, tw), -1, cv2.LINE_AA)
            if radius >= 2:
                cv2.circle(layer, (px, py), 1, (255, 255, 235), -1, cv2.LINE_AA)

    # latitude rings (dim wire grid)
    for ring in LAT_RINGS:
        poly(layer, ring, R, camd, col(C_MID, 0.30), T1)

    # data-dot shell with scan wave
    shell = SHELL @ rot_y(t * 0.15).T
    x, y, s, z = project(shell, R, camd)
    wave = 0.5 + 0.5 * np.sin(shell[:, 1] * 4.0 - t * 2.2)
    front = np.clip(0.5 - z / 2.0, 0.0, 1.0)
    for xi, yi, fr, wv in zip(x.tolist(), y.tolist(), front.tolist(), wave.tolist()):
        b = (0.15 + 0.85 * fr) * (0.40 + 0.60 * wv)
        r = 2 if (wv > 0.92 and fr > 0.5) else 1
        c = col(mix(C_BRIGHT, C_WHITE, 0.6 if r == 2 else 0.2), b)
        cv2.circle(layer, (int(xi), int(yi)), max(1, int(r * K)), c, -1, cv2.LINE_AA)

    # Gyroscope outer rings: thin, glowing, spinning on different axes, with comets
    for rad, tx, tz, spin, cspeed, tail in GYROS:
        Rr = rot_y(spin * t) @ rot_x(tx) @ rot_z(tz)
        pts = (UNIT_CIRCLE * rad) @ Rr.T
        shaded_ring(layer, glow_layer, pts, R, camd, C_BRIGHT, T1, GTH, 0.30, 1.0,
                    t=t, speed=cspeed, tail=tail)

    # Equator tick ring (slightly tilted)
    tick_ring(layer, glow_layer, R, camd, 1.08, 120, t, rot_z(0.22), 0.10, C_BRIGHT, 0.2, 0.95)

    # Orbits with travelling light
    for i, (pts, sp) in enumerate(ORBITS):
        po = pts @ rot_y(t * 0.07 * (1 + i % 3)).T
        shaded_ring(layer, glow_layer, po, R, camd, C_MID, T1, 3, 0.15, 0.55,
                    t=t + i * 0.37, speed=sp, tail=0.14)

    # ---- centre energy beat ----
    beat = 0.5 + 0.5 * math.sin(t * 3.4)
    beat = beat ** 1.3
    pulse = 0.5 + 0.9 * beat                 # 0.5 .. 1.4
    core_k = 0.7 + 0.6 * beat                # wireframe core size

    # core (spinning wireframe)
    Rc = rot_y(t * 0.6) @ rot_x(t * 0.35)
    for V, E, sc, th, bb in ((CORE_V, CORE_E, 0.20, T1, 0.85), (INNER_V, INNER_E, 0.11, T1, 0.6)):
        x, y, s, _ = project((V * sc * core_k) @ Rc.T, R, camd)
        for a, b2 in E:
            cv2.line(layer, (int(x[a]), int(y[a])), (int(x[b2]), int(y[b2])),
                     col(C_BRIGHT, bb), th, cv2.LINE_AA)
        for i in range(len(V)):
            cv2.circle(layer, (int(x[i]), int(y[i])), max(1, int(2.5 * K * s[i] / base)),
                       col((255, 255, 220)), -1, cv2.LINE_AA)

    cx, cy, cs, _ = project(np.zeros((1, 3)), R, camd)
    center = (int(cx[0]), int(cy[0]))
    crystal_r = max(5, int(0.15 * cs[0] * pulse))
    # Pulsing energy rings around the crystal
    for mult, b_, cc in ((2.8, 0.72, C_MID), (1.45, 0.92, C_BRIGHT)):
        rr = max(int(crystal_r * mult), 18 if mult > 2 else 10)
        cv2.circle(layer, center, rr, col(cc, b_ * pulse), max(1, T1), cv2.LINE_AA)
        cv2.circle(glow_layer, center, rr, col(cc, b_ * pulse), max(3, T2 * 2), cv2.LINE_AA)
    cv2.circle(glow_layer, center, int(crystal_r * (0.8 + 0.9 * beat)),
               col(C_BRIGHT, 0.12 + 0.30 * beat), -1, cv2.LINE_AA)
    # shockwave rings: expand from the centre and fade out
    for k in range(3):
        ph = (t * 0.8 + k / 3.0) % 1.0
        rr = int(crystal_r * (1.4 + ph * 6.0))
        bw = (1.0 - ph) ** 1.6
        cv2.circle(layer, center, rr, col(mix(C_BRIGHT, C_WHITE, 0.4), 0.9 * bw), 1, cv2.LINE_AA)
        cv2.circle(glow_layer, center, rr, col(C_BRIGHT, 0.9 * bw), max(3, T2 * 2), cv2.LINE_AA)

    crystal_rot = rot_y(t * 0.8) @ rot_x(t * 0.35)
    crystal_points = (CRYSTAL_V * (0.15 * pulse)) @ crystal_rot.T
    qx, qy, qs, qz = project(crystal_points, R, camd)
    face_layer = np.zeros_like(layer)
    face_order = sorted(range(len(CRYSTAL_F)),
                        key=lambda i: sum(qz[v] for v in CRYSTAL_F[i]), reverse=True)
    for face_i in face_order:
        face = CRYSTAL_F[face_i]
        pts = np.array([(int(qx[v]), int(qy[v])) for v in face], dtype=np.int32)
        shade = 0.24 if face_i % 2 == 0 else 0.14
        cv2.fillConvexPoly(face_layer, pts, col(C_BRIGHT, shade * pulse), cv2.LINE_AA)
    cv2.addWeighted(layer, 1.0, face_layer, 0.72, 0, dst=layer)
    for a, b2 in CRYSTAL_E:
        p1, p2 = (int(qx[a]), int(qy[a])), (int(qx[b2]), int(qy[b2]))
        cv2.line(layer, p1, p2, col(C_WHITE, 1.0 * pulse), max(1, round(K)), cv2.LINE_AA)
        cv2.line(glow_layer, p1, p2, col(C_BRIGHT, 1.0 * pulse), max(2, round(2 * K)), cv2.LINE_AA)
    highlight = (int(qx[0]), int(qy[0]))
    cv2.circle(layer, highlight, max(1, int(0.045 * crystal_r)), col((255, 255, 235)), -1, cv2.LINE_AA)

    nx, ny, ns, nz = project(NODES, R, camd)
    # glowing nodes
    for xi, yi, si, zi in zip(nx, ny, ns, nz):
        if -50 < xi < W + 50 and -50 < yi < H + 50:
            r = int(min(16, max(1, 1.35 * K * si / base)))
            front = 0.72 + 0.28 * (0.5 - zi / 2)
            px, py = int(xi), int(yi)
            cv2.circle(glow_layer, (px, py), r * 2, col(C_BRIGHT, front), -1, cv2.LINE_AA)
            cv2.circle(layer, (px, py), r, col(C_BRIGHT, front), -1, cv2.LINE_AA)
            cv2.circle(layer, (px, py), 1, (255, 255, 235), -1, cv2.LINE_AA)

    # Floating hologram words
    x, y, s, cz = project(LAB_POS, R, camd)
    for i in range(len(LAB_TXT)):
        xi, yi = x[i], y[i]
        if xi < -250 or xi > W + 250 or yi < -100 or yi > H + 100:
            continue
        fs = 0.22 * K * s[i] / base
        if fs < 0.14 or fs > 4:
            continue
        front = 0.5 - cz[i] / 2
        b = 0.45 + 0.55 * max(0.0, min(1.0, front))
        tw = 0.8 + 0.2 * math.sin(t * 3.0 + i * 1.3)       # each word shimmers slightly
        b *= tw
        th = 1 if fs < 1.2 else (2 if fs < 2.5 else 3)
        pos = (int(xi), int(yi))
        # big neon halo (glow layer) - 2 passes: thick + thin
        cv2.putText(glow_layer, LAB_TXT[i], pos, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    col(C_BRIGHT, 1.25 * b), th + 4, cv2.LINE_AA)
        cv2.putText(glow_layer, LAB_TXT[i], pos, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    col(C_BRIGHT, 1.4 * b), th + 1, cv2.LINE_AA)
        # bright white-hot core text
        cv2.putText(layer, LAB_TXT[i], pos, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    col(mix(C_TEXT, C_WHITE, 0.55), min(1.4, b * 1.35)), th, cv2.LINE_AA)

    # HUD arcs around the sphere
    draw_hud(layer, glow_layer, center, cs[0], t, zoom)
    draw_title(layer, glow_layer, t)

    # ---- bloom ----
    src = cv2.addWeighted(glow_layer, 1.0, layer, 0.45, 0)
    small = cv2.resize(src, (W // 3, H // 3), interpolation=cv2.INTER_AREA)
    tight = cv2.GaussianBlur(small, (0, 0), 1.2 * K)    # bright edge bloom
    middle = cv2.GaussianBlur(small, (0, 0), 3.7 * K)   # luminous local halo
    wide = cv2.GaussianBlur(small, (0, 0), 9.5 * K)     # soft atmospheric glow
    gk = GLOW[0]
    bloom = cv2.addWeighted(tight, 0.95 * gk, middle, 1.15 * gk, 0)
    bloom = cv2.addWeighted(bloom, 1.0, wide, 0.9 * gk, 0)
    glow = cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(layer, 1.18, glow, 0.66, 0)
    out = cv2.add(out, get_bg(RED_K[0]))
    out[::4] = (out[::4] * 0.94).astype(np.uint8)       # very faint scanlines
    if random.random() < 0.05:                          # slight hologram flicker
        out = cv2.convertScaleAbs(out, alpha=0.93)
    return out


# ------------------------------------------------------------------ hand tracking
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def download_model(path):
    """python.org Python on macOS often fails with CERTIFICATE_VERIFY_FAILED -> try 3 methods."""
    tmp = path + ".part"
    tries = []

    def plain():
        urllib.request.urlretrieve(MODEL_URL, tmp)

    def with_certifi():
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)

    def with_curl():                                   # curl is always available on macOS
        subprocess.run(["curl", "-L", "-f", "-s", "-o", tmp, MODEL_URL], check=True, timeout=120)

    for fn in (plain, with_certifi, with_curl):
        try:
            fn()
            if os.path.exists(tmp) and os.path.getsize(tmp) > 1_000_000:
                os.replace(tmp, path)
                return
        except Exception as e:
            tries.append(f"{fn.__name__}: {e!r}")
    raise RuntimeError("model download failed: " + " | ".join(tries))


def get_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if os.path.exists(path) and os.path.getsize(path) < 1_000_000:   # incomplete download
        os.remove(path)
    if not os.path.exists(path):
        print("Downloading hand model (~8 MB)...")
        download_model(path)
    return path


def open_camera():
    """Open a webcam on the main thread before creating any OpenCV window."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        avfoundation = getattr(cv2, "CAP_AVFOUNDATION", None)
        backends = [cv2.CAP_ANY] if avfoundation is None else [avfoundation, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for idx in range(5):
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                cap.release()
                continue
            # Give the OS time to grant access and deliver the first frame.
            t_end = time.time() + 5.0
            while time.time() < t_end:
                ok, frm = cap.read()
                if ok and frm is not None and frm.size:
                    print(f"Camera found (index {idx}, backend {be})")
                    return cap
                time.sleep(0.05)
            cap.release()
    return None


class Hands:
    """Camera opens on the MAIN thread (before the window). Frame reading + MediaPipe run in a thread."""

    def __init__(self):
        self.ok = False
        self.status = "OPENING CAMERA..."
        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.n = 0
        self.acc_dx = self.acc_dy = 0.0
        self.acc_zoom = 1.0
        self.prev_palm = None
        self.prev_dist = None
        self.palm_s = None
        self.detect = None
        print("Opening camera...")
        self.cap = open_camera()
        if self.cap is None:
            self.status = "CAMERA NOT FOUND"
            print("Camera could not be opened (hand tracking disabled; mouse control still works). "
                  "On macOS: System Settings > Privacy & Security > Camera -> enable your terminal/IDE, "
                  "then fully quit and reopen it. Close apps like Zoom/FaceTime that may be using the camera.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ok = True
        self.status = "LOADING HAND MODEL..."
        threading.Thread(target=self._run, daemon=True).start()

    def _load_model(self):
        if not HAS_MP:
            self.status = "MEDIAPIPE NOT INSTALLED"
            print("mediapipe is not installed (hand tracking disabled):  pip install -U mediapipe")
            return False
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model = get_model()
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.38, min_hand_presence_confidence=0.38,
                min_tracking_confidence=0.42)
            lm = vision.HandLandmarker.create_from_options(opts)
            self._t0 = time.time()
            self._last_ts = 0

            def det(rgb):
                ts = max(self._last_ts + 1, int((time.time() - self._t0) * 1000))
                self._last_ts = ts
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                return res.hand_landmarks
            self.detect = det
            return True
        except Exception as e:
            self.status = "HAND MODEL ERROR (see terminal)"
            print("Hand tracking failed to start:", repr(e))
            print("Download the model manually and place it next to interface.py:\n  ", MODEL_URL)
            return False

    def _run(self):
        # Keep reading frames immediately; model download/init can take several seconds.
        threading.Thread(target=self._load_model_background, daemon=True).start()
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                frame = cv2.flip(frame, 1)
                if self.detect is not None:
                    self._process(frame)
                else:
                    with self.lock:
                        self.frame = frame
        except Exception as e:
            self.status = "TRACKER CRASHED (see terminal)"
            print("Tracker error:", repr(e))

    def _load_model_background(self):
        try:
            if self._load_model():
                self.status = "SHOW HANDS"
        except Exception as e:
            self.status = "HAND MODEL ERROR (see terminal)"
            print("Hand model load error:", repr(e))

    @staticmethod
    def d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _process(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands = self.detect(rgb)
        fh, fw = frame.shape[:2]
        info = []
        for lm in hands:
            size = max(self.d(lm[0], lm[9]), 1e-4)
            pinch = self.d(lm[4], lm[8]) / size < 0.55           # loose threshold = easy pinch
            pp = ((lm[4].x + lm[8].x) / 2, (lm[4].y + lm[8].y) / 2)
            info.append((pinch, (lm[9].x, lm[9].y), pp))
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, (int(lm[a].x * fw), int(lm[a].y * fh)),
                         (int(lm[b].x * fw), int(lm[b].y * fh)), (255, 255, 255), 2)
            for p in lm:
                cv2.circle(frame, (int(p.x * fw), int(p.y * fh)), 3, (255, 255, 255), -1)
            if pinch:
                cv2.circle(frame, (int(pp[0] * fw), int(pp[1] * fh)), 10, (0, 255, 255), 2)

        dx = dy = 0.0
        zf = 1.0
        if len(info) == 2 and info[0][0] and info[1][0]:          # both pinching -> zoom
            dist = math.hypot(info[0][2][0] - info[1][2][0], info[0][2][1] - info[1][2][1])
            if self.prev_dist:
                zf = max(0.7, min(1.4, (dist / self.prev_dist) ** 1.8))
            self.prev_dist = dist
            self.prev_palm = None
            self.palm_s = None
        else:
            self.prev_dist = None
            if len(info) == 1:                                    # one hand -> spin
                p = info[0][1]
                self.palm_s = p if self.palm_s is None else (
                    self.palm_s[0] * 0.5 + p[0] * 0.5, self.palm_s[1] * 0.5 + p[1] * 0.5)
                if self.prev_palm is not None:
                    dx = self.palm_s[0] - self.prev_palm[0]
                    dy = self.palm_s[1] - self.prev_palm[1]
                    if abs(dx) < 0.0015 and abs(dy) < 0.0015:     # ignore hand jitter
                        dx = dy = 0.0
                self.prev_palm = self.palm_s
            else:
                self.prev_palm = None
                self.palm_s = None

        with self.lock:
            self.frame = frame
            self.n = len(info)
            self.acc_dx += dx
            self.acc_dy += dy
            self.acc_zoom *= zf
            self.status = "SHOW HANDS" if not info else f"{len(info)} HAND(S)"

    def poll(self):
        """returns (preview_frame, (dx, dy), zoom_factor, n_hands) accumulated since the last call"""
        with self.lock:
            out = (None if self.frame is None else self.frame.copy(),
                   (self.acc_dx, self.acc_dy), self.acc_zoom, self.n)
            self.acc_dx = self.acc_dy = 0.0
            self.acc_zoom = 1.0
        return out

    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.cap is not None:
            self.cap.release()


# ------------------------------------------------------------------ app state / UI
S = dict(yaw=0.0, pitch=0.15, vyaw=AUTO_SPEED, vpitch=0.0, zoom=1.0, zoom_t=1.0,
         drag=False, lx=0, ly=0, mdx=0.0, mdy=0.0)


def clampz(z):
    return max(ZOOM_MIN, min(ZOOM_MAX, z))


def press(name):
    if name == "+":
        S["zoom_t"] = clampz(S["zoom_t"] * 1.25)
    elif name == "-":
        S["zoom_t"] = clampz(S["zoom_t"] / 1.25)
    else:
        S.update(yaw=0.0, pitch=0.15, zoom_t=1.0, vyaw=AUTO_SPEED, vpitch=0.0)


def on_mouse(ev, x, y, flags, _):
    if ev == cv2.EVENT_LBUTTONDOWN:
        S["drag"], S["lx"], S["ly"] = True, x, y
    elif ev == cv2.EVENT_MOUSEMOVE and S["drag"]:
        S["mdx"] += (x - S["lx"]) / W
        S["mdy"] += (y - S["ly"]) / W
        S["lx"], S["ly"] = x, y
    elif ev == cv2.EVENT_LBUTTONUP:
        S["drag"] = False
    elif ev == cv2.EVENT_MOUSEWHEEL:
        S["zoom_t"] = clampz(S["zoom_t"] * (1.1 if flags > 0 else 1 / 1.1))


def draw_scanline(img, t):
    """Transparent beam with a soft halo, moving smoothly up and down."""
    span = max(1, H - 1)
    phase = (t * 190) % (2 * span)
    y_float = phase if phase <= span else 2 * span - phase
    y = int(y_float)
    frac = y_float - y
    tint = np.empty((1, W, 3), dtype=np.uint8)
    tint[:] = C_BRIGHT
    for offset, alpha in ((-3, 0.035), (-2, 0.07), (-1, 0.16),
                          (0, 0.68), (1, 0.16), (2, 0.07), (3, 0.035)):
        for yy, weight in ((y + offset, 1.0 - frac), (y + offset + 1, frac)):
            if 0 <= yy < H and weight > 0.001:
                row = img[yy:yy + 1]
                a = alpha * weight
                img[yy:yy + 1] = cv2.addWeighted(row, 1.0 - a, tint, a, 0)


def sync_render_size():
    """Match the render canvas to the resizable window (capped to save power)."""
    global W, H, K, F
    try:
        _, _, view_w, view_h = cv2.getWindowImageRect(WIN)
    except cv2.error:
        return
    if view_w < 320 or view_h < 240:
        return
    scale = min(1.0, 1600 / view_w, 1000 / view_h)
    new_w = max(320, int(view_w * scale))
    new_h = max(240, int(view_h * scale))
    if new_w != W or new_h != H:
        W, H = new_w, new_h
        K = H / 720.0
        F = 1.5 * H


def main():
    tracker = Hands()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.setMouseCallback(WIN, on_mouse)
    full = False                       # press F to toggle fullscreen
    t0 = last = time.time()
    while True:
        frame_start = time.time()
        now = frame_start
        dt = max(1e-3, min(0.1, now - last))
        last = now

        cam, rot, zf, nh = tracker.poll()
        if abs(zf - 1.0) > 1e-4:
            S["zoom_t"] = clampz(S["zoom_t"] * zf)

        dx, dy = rot
        dx += S["mdx"]
        dy += S["mdy"]
        S["mdx"] = S["mdy"] = 0.0

        if abs(dx) > 1e-5 or abs(dy) > 1e-5:         # manual spin
            a_y, a_p = -dx * 6.0, -dy * 5.0
            S["yaw"] += a_y
            S["pitch"] += a_p
            S["vyaw"] = 0.6 * S["vyaw"] + 0.4 * (a_y / dt)
            S["vpitch"] = 0.6 * S["vpitch"] + 0.4 * (a_p / dt)
        else:                                         # inertia -> back to slow auto rotation
            k = min(1.0, dt * 1.5)
            S["vyaw"] += (AUTO_SPEED - S["vyaw"]) * k
            S["vpitch"] += (0.0 - S["vpitch"]) * k
            S["yaw"] += S["vyaw"] * dt
            S["pitch"] += S["vpitch"] * dt
        S["pitch"] = max(-1.4, min(1.4, S["pitch"]))
        S["zoom"] += (S["zoom_t"] - S["zoom"]) * min(1.0, dt * 8)

        # Red (smooth fade) while Vision speaks, blue otherwise
        target = 1.0 if vision_speaking() else 0.0
        if abs(target - RED_K[0]) > 0.002:
            RED_K[0] += (target - RED_K[0]) * min(1.0, dt * 7.0)
            set_palette(RED_K[0])
        elif RED_K[0] != target:
            RED_K[0] = target
            set_palette(target)

        img = render(S["yaw"], S["pitch"], S["zoom"], now - t0)
        draw_scanline(img, now - t0)
        cv2.imshow(WIN, img)

        k = cv2.waitKey(1) & 0xFF
        sync_render_size()
        if k in (27, ord("q")):
            break
        elif k in (ord("+"), ord("=")):
            press("+")
        elif k in (ord("-"), ord("_")):
            press("-")
        elif k == ord("]"):
            GLOW[0] = min(3.0, GLOW[0] * 1.2)
        elif k == ord("["):
            GLOW[0] = max(0.2, GLOW[0] / 1.2)
        elif k == ord("r"):
            press("RESET")
        elif k == ord("f"):
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        # Note: WND_PROP_VISIBLE can report 0 spuriously on macOS, so Q/Esc is the shutdown path.
        time.sleep(max(0.0, 1.0 / 45.0 - (time.time() - frame_start)))

    tracker.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
