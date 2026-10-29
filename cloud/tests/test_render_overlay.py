"""The rendered video draws what the player draws (memory/61).

``draw_frame`` is driven by the web's render spec and the web's own STYLE
(loaded from beemonitor_web/apps/analysis/overlay_spec.py), so these tests also
hold the two to one contract.
"""

import gzip
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sagemaker_backend"))

import render_overlay as ro  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "overlay_spec", ROOT / "beemonitor_web" / "apps" / "analysis" / "overlay_spec.py")
overlay_spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(overlay_spec)
STYLE = overlay_spec.STYLE

W, H, FPS = 320, 240, 10.0
OFF = {k: False for k in overlay_spec.LAYERS}


def spec(rows=(), events=(), interactions=(), regions=(), layers=None, rotate=False,
         clip_start="", tracks=None):
    flat = [v for r in rows for v in r]
    return {"v": 1, "fps": FPS, "rotate_180": rotate, "clip_start": clip_start,
            "style": STYLE, "layers": {**OFF, **(layers or {})},
            "tracks": tracks or {}, "rows": flat, "row_width": 7,
            "events": list(events), "interactions": list(interactions), "regions": list(regions)}


def render(s, frame_no):
    frame = np.zeros((H, W, 3), np.uint8)
    return ro.draw_frame(frame, frame_no, s, ro.SpecIndex(s, W, H))


def near(px, bgr, tol=40):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(px, bgr))


BOX = (100, 100, 160, 150)
REF3 = {"label": "reference_3", "kind": "reference",
        "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]}   # px (32,24)-(96,72)


def test_colour_is_the_players_hsl_rule():
    assert ro.track_bgr(12, STYLE) == tuple(reversed(overlay_spec.track_rgb(12)))


def test_a_box_is_drawn_in_its_track_colour():
    img = render(spec(rows=[(5, 12, *BOX, 0)], layers={"boxes": True}), 5)
    assert near(img[125, 100], ro.track_bgr(12, STYLE))    # left edge, below the label


def test_a_lost_box_is_dashed():
    img = render(spec(rows=[(5, 12, *BOX, 1)], layers={"boxes": True, "lost": True}), 5)
    edge = img[110:150, 160].max(axis=1)                     # right edge, below any label
    assert (edge > 30).any() and (edge < 5).any()


def test_layers_off_draw_nothing():
    s = spec(rows=[(5, 12, *BOX, 0)], regions=[REF3],
             events=[[5, 12, "exit", "reference_3", None, False]])
    assert render(s, 5).max() == 0


def test_an_exit_chip_shows_for_event_seconds_after_and_not_after():
    exit_bgr = ro.rgba_bgr(STYLE["event_rgba"]["exit"])[0]
    s = spec(rows=[(f, 12, *BOX, 0) for f in range(0, 40)], regions=[REF3],
             events=[[10, 12, "exit", "reference_3", None, False]], layers={"events": True})

    def has_exit(img):
        return bool(np.all(np.abs(img.astype(int) - exit_bgr) <= 3, axis=2).sum() > 20)
    assert has_exit(render(s, 10))
    assert has_exit(render(s, 10 + int(STYLE["event_seconds"] * FPS)))
    assert not has_exit(render(s, 10 + int(STYLE["event_seconds"] * FPS) + 2))
    assert not has_exit(render(s, 9))


def test_an_enter_chip_shows_before_the_enter():
    enter_bgr = ro.rgba_bgr(STYLE["event_rgba"]["enter"])[0]
    s = spec(rows=[(f, 12, *BOX, 0) for f in range(0, 30)], regions=[REF3],
             events=[[29, 12, "enter", "reference_3", None, None]], layers={"events": True})

    def has_enter(img):
        return bool(np.all(np.abs(img.astype(int) - enter_bgr) <= 3, axis=2).sum() > 20)
    assert has_enter(render(s, 20))
    assert not has_enter(render(s, 10))


def test_in_reference_restrokes_the_reference_in_the_track_colour():
    s = spec(rows=[(5, 12, 40, 40, 60, 60, 0)], regions=[REF3],
             interactions=[[0, 10, 12, "reference_3", "in_reference", None]],
             layers={"interactions": True})
    img = render(s, 5)
    assert near(img[48, 32], ro.track_bgr(12, STYLE))       # the reference's left edge


def test_near_draws_a_line_between_the_two():
    near_bgr = ro.rgba_bgr(STYLE["near_rgba"])[0]
    s = spec(rows=[(5, 1, 20, 200, 30, 210, 0), (5, 2, 280, 200, 290, 210, 0)],
             interactions=[[0, 10, 1, 2, "near", 1.4]], layers={"interactions": True})
    img = render(s, 5)
    assert near(img[205, 100], near_bgr)


def test_rotate_180_moves_the_box_to_the_turned_position():
    s = spec(rows=[(5, 12, *BOX, 0)], layers={"boxes": True}, rotate=True)
    img = render(s, 5)
    x1, y1, x2, y2 = W - BOX[2], H - BOX[3], W - BOX[0], H - BOX[1]
    assert near(img[(y1 + y2) // 2, x2], ro.track_bgr(12, STYLE))
    assert img[(BOX[1] + BOX[3]) // 2, BOX[0]].max() == 0     # nothing where it was unturned


def test_clock_draws_only_when_on():
    s = spec(layers={"clock": True}, clip_start="2026-06-01T10:00:00.000-04:00")
    assert render(s, 12).max() > 0
    assert render(spec(), 12).max() == 0


def test_annotate_needs_the_render_spec():
    import importlib.util as iu
    sp = iu.spec_from_file_location("sm_inference_r", ROOT / "sagemaker_backend" / "inference.py")
    inference = iu.module_from_spec(sp)
    sp.loader.exec_module(inference)
    out = inference._annotate_video({"job_id": "a", "user_id": 1, "video_blob_path": "v"},
                                    mock.Mock())
    assert out["status"] == "failed" and "render_spec_path" in out["error_message"]


def test_annotate_renders_from_the_spec(tmp_path):
    import cv2

    src = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(12):
        vw.write(np.zeros((H, W, 3), np.uint8))
    vw.release()
    body = gzip.compress(json.dumps(spec(rows=[(f, 3, *BOX, 0) for f in range(12)],
                                         layers={"boxes": True})).encode())
    uploaded = {}

    class Storage:
        def download_to_stream(self, bucket, key, buf):
            buf.write(body)

        def download_file(self, bucket, key, dest):
            Path(dest).write_bytes(src.read_bytes())

        def upload_file(self, bucket, key, path, content_type=None):
            uploaded[key] = Path(path).stat().st_size

    import importlib.util as iu
    sp = iu.spec_from_file_location("sm_inference_r2", ROOT / "sagemaker_backend" / "inference.py")
    inference = iu.module_from_spec(sp)
    sp.loader.exec_module(inference)
    pipeline = mock.Mock(_storage=Storage())
    out = inference._annotate_video({"job_id": "annot_x", "user_id": 7, "video_blob_path": "v",
                                     "render_spec_path": "specs/x.json.gz"}, pipeline)
    assert out["status"] == "completed", out
    assert out["frames"] == 12
    assert uploaded.get("7/annot_x/annotated.mp4", 0) > 0


def test_label_strings_are_the_players():
    assert ro.label_text(12, "Osmia lignaria") == "12 · Osmia lignaria"
    assert ro.label_text(12, lost_seconds=1.23) == "12 · lost 1.2 s"
    assert ro.label_text(12, in_reference="reference_3", in_seconds=2.3) == "12 · in 3 · 2.3 s"
    assert ro.label_text(4, "Ceratina dupla", 0.5, "reference_5", 1.0) == \
        "4 · Ceratina dupla · lost 0.5 s · in 5 · 1.0 s"
    assert ro.event_text("exit", "reference_3", True) == "EXIT 3 · at edge"
    assert ro.event_text("enter", "reference_3", None) == "ENTER 3"
    assert ro.near_text(1.44) == "near 1.4 bl"
    assert ro.near_text(None) == "near"
    from datetime import datetime
    assert ro.clock_text(72.4, datetime.fromisoformat("2026-06-01T10:00:00.000-04:00")) == \
        "01:12.4  2026-06-01 10:01:12.4"


def test_a_non_ascii_name_draws_as_a_sprite():
    sprite = ro._sprite("12 · Ōsmia ñ", 12, True, (17, 24, 39, 255), (200, 120, 40, 242), 4, 16)
    assert sprite.ndim == 3 and sprite.shape[2] == 4
    rgb = sprite[:, :, :3].reshape(-1, 3)
    assert len({tuple(p) for p in rgb}) > 2               # text pixels, not a plain fill


def test_the_label_sprite_lands_on_the_frame():
    s = spec(rows=[(5, 12, *BOX, 0)], layers={"boxes": True, "species": True},
             tracks={"12": {"species": "Osmia lignaria"}})
    img = render(s, 5)
    chip = img[BOX[1] - 16:BOX[1] - 1, BOX[0]:BOX[0] + 40]
    assert near(chip[1, 1], ro.track_bgr(12, STYLE), tol=60)   # the filled chip
    assert (np.abs(chip.astype(int) - chip[1, 1].astype(int)).sum(axis=2) > 120).any()  # its text
    assert ro._sprite.cache_info().currsize > 0


def test_sprites_are_drawn_once_and_reused():
    ro._sprite.cache_clear()
    s = spec(rows=[(f, 12, *BOX, 0) for f in range(10)], layers={"boxes": True})
    for f in range(10):
        render(s, f)
    info = ro._sprite.cache_info()
    assert info.misses == 1 and info.hits >= 9
