"""Draw the player's overlay onto video frames (memory/61).

The rendered video and the browser player are two renderers of one picture.
Everything drawn here comes from the RENDER SPEC the web builds (boxes, track
identities, events, interactions, ROI and references, the layers that were on,
and the style); nothing is re-derived on the worker, so the file can't disagree
with the player or with the run's CSVs. The drawing rules are the docstring of
``beemonitor_web/apps/analysis/overlay_spec.py``; the numbers are its ``STYLE``,
carried in the spec.

``draw_frame`` is pure (numpy + OpenCV) so it can be tested without S3 or
ffmpeg; ``inference._annotate_video`` streams frames through it.
"""

from __future__ import annotations

import colorsys
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# The player's separator, drawn as is: text is set with Pillow (DejaVu Sans
# Mono), not OpenCV's ASCII-only Hershey fonts, so "·" and species names with
# accents come out exactly as on the page.
SEP = " · "
FONT_PATHS = {
    False: "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    True: "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
}
_REF_NUMBER = re.compile(r"^(?:reference|nest|tube|region)[ _-]?(\d+)$", re.IGNORECASE)


# ── Colours ──────────────────────────────────────────────────────────────────

def track_bgr(track_id, style):
    """HSL(id × hue_step mod 360, sat %, light %) as BGR — the player's rule."""
    hue = (float(track_id) * style["hue_step"]) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, style["light"] / 100, style["sat"] / 100)
    return (round(b * 255), round(g * 255), round(r * 255))


def rgba_bgr(rgba):
    """``[r, g, b, a]`` as ``((b, g, r), a)``."""
    r, g, b = rgba[:3]
    a = rgba[3] if len(rgba) > 3 else 1.0
    return (int(b), int(g), int(r)), float(a)


def reference_text(label):
    """"reference_3" -> "3"; "ROI" and names a user typed stay whole."""
    m = _REF_NUMBER.match(str(label or ""))
    return m.group(1) if m else str(label or "")


# ── Text: Pillow sprites, cached, alpha-blitted ──────────────────────────────

@lru_cache(maxsize=64)
def _font(px, bold):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(FONT_PATHS[bold], px)
    except OSError:
        logger.warning("render_overlay: %s missing — using Pillow's default font",
                       FONT_PATHS[bold])
        return ImageFont.load_default(size=px)


@lru_cache(maxsize=4096)
def _sprite(text, px, bold, fg, bg, pad, height):
    """A chip — ``bg`` (RGBA) filled, ``text`` in ``fg`` (RGBA) — as a BGRA array.
    Each distinct chip is drawn once; a clip's labels repeat frame after frame."""
    from PIL import Image, ImageDraw

    font = _font(px, bold)
    left, top, right, bottom = font.getbbox(text)
    tw, th = int(right - left), int(bottom - top)
    w = tw + 2 * pad
    h = max(int(height or 0), th + 2 * pad)
    img = Image.new("RGBA", (max(w, 1), max(h, 1)), tuple(bg))
    ImageDraw.Draw(img).text((pad - left, (h - th) // 2 - top), text, font=font, fill=tuple(fg))
    rgba = np.asarray(img)
    return rgba[:, :, [2, 1, 0, 3]].copy()


def _blit(frame, sprite, x, y):
    """Alpha-composite a BGRA sprite onto the frame at (x, y), clipped."""
    h, w = sprite.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sp = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sp[:, :, 3:4].astype(np.float32) / 255.0
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (sp[:, :, :3] * a + region * (1 - a)).astype(frame.dtype)


def _rgba255(bgr, alpha):
    b, g, r = bgr
    return (int(r), int(g), int(b), int(round(alpha * 255)))


def text_width(text, px, bold=False):
    return _font(int(round(px)), bold).getlength(text)


# ── The strings the player draws ─────────────────────────────────────────────

def label_text(track_id, species=None, lost_seconds=None, in_reference=None, in_seconds=None):
    """"12 · Osmia lignaria · lost 1.2 s · in 3 · 2.3 s" — as the player."""
    text = str(track_id)
    if species:
        text += SEP + str(species)
    if lost_seconds is not None:
        text += SEP + f"lost {lost_seconds:.1f} s"
    if in_reference is not None:
        text += SEP + f"in {reference_text(in_reference)}" + SEP + f"{in_seconds:.1f} s"
    return text


def event_text(action, reference, at_edge=None):
    text = f"{action.upper()} {reference_text(reference)}"
    return text + SEP + "at edge" if at_edge is True else text


def near_text(min_bl):
    return "near" if min_bl is None else f"near {float(min_bl):.1f} bl"


def clock_text(seconds, clip_start=None):
    text = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
    if clip_start is not None:
        ts = clip_start + timedelta(seconds=seconds)
        text += "  " + ts.strftime("%Y-%m-%d %H:%M:%S.") + str(ts.microsecond // 100000)
    return text


# ── Index: built once per video ──────────────────────────────────────────────

class SpecIndex:
    """The spec reorganised for per-frame lookups, in output-frame pixels
    (``rotate_180`` already applied to every coordinate)."""

    def __init__(self, spec, width, height):
        self.spec = spec
        self.W, self.H = int(width), int(height)
        self.rotate = bool(spec.get("rotate_180"))
        self.fps = float(spec.get("fps") or 25.0) or 25.0
        self.style = spec["style"]
        self.layers = spec.get("layers") or {}
        self.scale = max(1.0, self.W / 1280.0)
        self.tracks = spec.get("tracks") or {}

        rows = spec.get("rows") or []
        width_ = int(spec.get("row_width") or 7)
        self.by_frame = {}
        per = {}
        for i in range(0, len(rows) - width_ + 1, width_):
            f, tid, x1, y1, x2, y2, lost = rows[i:i + 7]
            x1, y1, x2, y2 = self.box((x1, y1, x2, y2))
            self.by_frame.setdefault(int(f), []).append((int(tid), x1, y1, x2, y2, int(lost)))
            t = per.setdefault(int(tid), ([], [], [], []))
            t[0].append(int(f)); t[1].append((x1 + x2) / 2); t[2].append((y1 + y2) / 2)
            t[3].append(int(lost))
        self.per_track = {tid: (np.array(f), np.array(cx, float), np.array(cy, float),
                                np.array(lo, np.int8))
                          for tid, (f, cx, cy, lo) in per.items()}

        self.regions = []
        for r in spec.get("regions") or []:
            pts = [self.point(p[0] * self.W, p[1] * self.H) for p in r.get("points") or []]
            if len(pts) >= 3:
                self.regions.append({"label": r.get("label", ""), "kind": r.get("kind", "reference"),
                                     "pts": np.array(pts, np.int32)})
        self.region_by_id = {r["label"]: r for r in self.regions if r["kind"] == "reference"}

        E = float(self.style["event_seconds"]) * self.fps
        self.events = []
        for frame, tid, action, ref, end, at_edge in spec.get("events") or []:
            frame = int(frame)
            if action == "exit":
                lo, hi = frame, frame + E
            elif action == "enter":
                lo, hi = frame - E, frame
            else:
                lo, hi = frame, int(end) if end is not None else frame
            self.events.append({"lo": lo, "hi": hi, "track": int(tid), "action": action,
                                "reference": ref, "at_edge": at_edge})
        self.interactions = []
        for start, end, tid, other, relation, min_bl in spec.get("interactions") or []:
            self.interactions.append({"start": int(start), "end": int(end), "track": int(tid),
                                      "with": other, "relation": relation, "min_bl": min_bl})

        start = spec.get("clip_start") or ""
        try:
            self.clip_start = datetime.fromisoformat(start) if start else None
        except ValueError:
            self.clip_start = None

    # Coordinates in the original frame -> the drawn (possibly turned) frame.
    def point(self, x, y):
        return (self.W - x, self.H - y) if self.rotate else (x, y)

    def box(self, b):
        x1, y1, x2, y2 = b
        if self.rotate:
            x1, y1, x2, y2 = self.W - x2, self.H - y2, self.W - x1, self.H - y1
        return x1, y1, x2, y2

    def px(self, value):
        return max(1, int(round(float(value) * self.scale)))

    def lost_seconds(self, tid, frame):
        t = self.per_track.get(tid)
        if t is None:
            return 0.0
        frames, _cx, _cy, lost = t
        i = int(np.searchsorted(frames, frame, side="right")) - 1
        while i >= 0 and lost[i]:
            i -= 1
        return 0.0 if i < 0 else (frame - frames[i]) / self.fps


# ── Primitives ───────────────────────────────────────────────────────────────

def _blend(frame, alpha, draw):
    """Run ``draw(img)`` at ``alpha`` over the frame."""
    if alpha >= 0.999:
        draw(frame)
        return
    layer = frame.copy()
    draw(layer)
    mask = np.any(layer != frame, axis=2)
    frame[mask] = (layer[mask] * alpha + frame[mask] * (1 - alpha)).astype(frame.dtype)


def _dashed_poly(img, pts, color, width, dash, closed=True):
    pts = [tuple(map(float, p)) for p in pts]
    if closed:
        pts = pts + [pts[0]]
    on, off = float(dash[0]), float(dash[1])
    period, carry = on + off, 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        seg = float(np.hypot(bx - ax, by - ay))
        if seg == 0:
            continue
        pos = 0.0
        while pos < seg:
            phase = (carry + pos) % period
            if phase < on:
                run = min(on - phase, seg - pos)
                s, e = pos / seg, (pos + run) / seg
                cv2.line(img, (int(round(ax + (bx - ax) * s)), int(round(ay + (by - ay) * s))),
                         (int(round(ax + (bx - ax) * e)), int(round(ay + (by - ay) * e))),
                         color, width, cv2.LINE_AA)
                pos += run
            else:
                pos += period - phase
        carry = (carry + seg) % period


def _chip(frame, text, x, y, px, bg_bgr, bg_alpha, fg_bgr, pad, height=None, bold=False,
          fg_alpha=1.0):
    """A filled chip with text, top-left at (x, y), kept inside the frame.
    Returns its (w, h)."""
    sprite = _sprite(str(text), int(round(px)), bool(bold), _rgba255(fg_bgr, fg_alpha),
                     _rgba255(bg_bgr, bg_alpha), int(pad), int(height or 0))
    h, w = sprite.shape[:2]
    x = int(max(0, min(x, frame.shape[1] - w)))
    y = int(max(0, min(y, frame.shape[0] - h)))
    _blit(frame, sprite, x, y)
    return w, h


# ── One frame ────────────────────────────────────────────────────────────────

def draw_frame(frame, frame_no, spec, index):
    """Draw the overlay for ``frame_no`` onto ``frame`` (BGR, already turned
    when the unit is mounted upside down). Returns the frame."""
    st, L, ix = index.style, index.layers, index
    rows = [r for r in ix.by_frame.get(frame_no, ()) if L.get("lost", True) or not r[5]]
    here = {r[0]: r for r in rows}

    # ROI and reference objects.
    if L.get("regions"):
        for r in ix.regions:
            color, alpha = rgba_bgr(st["roi_rgba"] if r["kind"] == "roi" else st["reference_rgba"])
            _blend(frame, alpha, lambda img, r=r, c=color: _dashed_poly(
                img, r["pts"], c, ix.px(st["region_width"]), [ix.px(d) for d in st["dash"]]))
            _region_label(frame, r, ix)

    # Interactions in a reference: the reference re-stroked in the insect's colour.
    active_in = {}
    if L.get("interactions"):
        for it in ix.interactions:
            if it["relation"] == "in_reference" and it["start"] <= frame_no <= it["end"]:
                active_in[it["track"]] = it
                ref = ix.region_by_id.get(str(it["with"]))
                if ref is not None:
                    cv2.polylines(frame, [ref["pts"]], True, track_bgr(it["track"], st),
                                  ix.px(st["in_reference_width"]), cv2.LINE_AA)

    # Events: the reference glows in the action's colour.
    events_now = []
    if L.get("events"):
        for ev in ix.events:
            if ev["lo"] <= frame_no <= ev["hi"]:
                events_now.append(ev)
                ref = ix.region_by_id.get(str(ev["reference"]))
                if ref is not None:
                    color, _a = rgba_bgr(st["event_rgba"][ev["action"]])
                    cv2.polylines(frame, [ref["pts"]], True, color, ix.px(st["glow_width"]), cv2.LINE_AA)

    # Trails.
    if L.get("trails"):
        for tid in here:
            _trail(frame, tid, frame_no, ix)

    # Boxes and their labels.
    font_px = st["label_font_px"] * ix.scale
    if L.get("boxes", True):
        for tid, x1, y1, x2, y2, lost in rows:
            color = track_bgr(tid, st)
            alpha = st["lost_alpha"] if lost else 1.0
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if lost:
                _blend(frame, alpha, lambda img, p=pts, c=color: _dashed_poly(
                    img, p, c, ix.px(st["box_width"]), [ix.px(d) for d in st["dash"]]))
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color,
                              ix.px(st["box_width"]), cv2.LINE_AA)
            track = ix.tracks.get(str(tid)) or {}
            spell = active_in.get(tid)
            text = label_text(
                tid,
                species=track.get("species") if L.get("species", True) else None,
                lost_seconds=ix.lost_seconds(tid, frame_no) if lost else None,
                in_reference=spell["with"] if spell is not None else None,
                in_seconds=(frame_no - spell["start"]) / ix.fps if spell is not None else None)
            th = int(round(st["label_height"] * ix.scale))
            ty = y1 - th - 1 if y1 - th - 1 >= 0 else y2 + 1
            # The player sets labels at weight 600: the bold face here.
            _chip(frame, text, x1, ty, font_px, color, 0.7 if lost else 0.95,
                  rgba_bgr(st["label_text_rgba"])[0], ix.px(st["label_pad"]), height=th, bold=True)

    # Event chips: under the insect, or at the reference when it isn't in view.
    for ev in events_now:
        color, _a = rgba_bgr(st["event_rgba"][ev["action"]])
        text = event_text(ev["action"], ev["reference"], ev["at_edge"])
        r = here.get(ev["track"])
        if r is not None:
            x, y = r[1], r[4] + int(round(st["label_height"] * ix.scale)) + 2
        else:
            ref = ix.region_by_id.get(str(ev["reference"]))
            if ref is None:
                continue
            x, y = int(ref["pts"][:, 0].min()), int(ref["pts"][:, 1].max()) + 2
        _chip(frame, text, x, y, st["chip_font_px"] * ix.scale, color, 1.0,
              rgba_bgr(st["chip_text_rgba"])[0], ix.px(st["label_pad"]), bold=True)

    # Insects near each other: a line between their centres.
    if L.get("interactions"):
        color, alpha = rgba_bgr(st["near_rgba"])
        for it in ix.interactions:
            if it["relation"] != "near" or not (it["start"] <= frame_no <= it["end"]):
                continue
            a, b = here.get(it["track"]), here.get(_int(it["with"]))
            if a is None or b is None:
                continue
            pa = (int((a[1] + a[3]) / 2), int((a[2] + a[4]) / 2))
            pb = (int((b[1] + b[3]) / 2), int((b[2] + b[4]) / 2))
            _blend(frame, alpha, lambda img: cv2.line(img, pa, pb, color, ix.px(st["near_width"]),
                                                      cv2.LINE_AA))
            text = near_text(it["min_bl"])
            _chip(frame, text, (pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2,
                  st["chip_font_px"] * ix.scale, (0, 0, 0), 0.65, color, ix.px(st["label_pad"]),
                  bold=True)

    if L.get("clock"):
        text = clock_text(frame_no / ix.fps, ix.clip_start)
        _chip(frame, text, ix.px(8), ix.px(8), st["chip_font_px"] * ix.scale, (0, 0, 0), 0.6,
              (255, 255, 255), ix.px(st["label_pad"]))
    return frame


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _trail(frame, tid, frame_no, ix):
    st = ix.style
    t = ix.per_track.get(tid)
    if t is None:
        return
    frames, cx, cy, _lost = t
    end = int(np.searchsorted(frames, frame_no, side="right"))
    start = int(np.searchsorted(frames, frame_no - round(st["trail_seconds"] * ix.fps), side="left"))
    if end - start < 2:
        return
    gap = st["trail_gap_seconds"] * ix.fps
    color = track_bgr(tid, st)
    width = max(1, int(round(st["trail_width"] * ix.scale)))

    def draw(img):
        for i in range(start + 1, end):
            if frames[i] - frames[i - 1] > gap:
                continue          # a gap over a second is two paths, not one
            cv2.line(img, (int(cx[i - 1]), int(cy[i - 1])), (int(cx[i]), int(cy[i])),
                     color, width, cv2.LINE_AA)
    _blend(frame, st["trail_alpha"], draw)


def _region_label(frame, region, ix):
    """The reference's number in a dark chip inside its top-left corner, shrunk
    to fit; left off when it can't fit."""
    st = ix.style
    text = reference_text(region["label"])
    if not text:
        return
    pts = region["pts"]
    x0, y0, x1 = int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max())
    px = st["region_label_px"] * ix.scale
    w = text_width(text, px)
    while w + 4 > x1 - x0 and px > 8 * ix.scale:
        px -= ix.scale
        w = text_width(text, px)
    if w + 4 > x1 - x0:
        return
    _chip(frame, text, x0 + 1, y0 + 1, px, (0, 0, 0), 0.55, (255, 255, 255), 2, fg_alpha=0.95)
