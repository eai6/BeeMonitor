"""Sample-and-pre-label: motion proposes, the detector confirms, only moving
detections count. Each test mirrors a finding from calibrating on real clips
(memory/38 §4.5): nest plugs a detector mistakes for bees, the burned-in
clock, a hand or light change filling the frame, and frame numbers that must
match the editor's."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from beemonitor.processing import sample_label as sl

W, H, FPS = 640, 360, 25
PLUGS = [(100, 250), (200, 250), (300, 250), (400, 250)]   # static dark squares
BEE_FRAMES = range(50, 90)
FLASH_FRAMES = range(120, 126)


def _write(path, n=150, bee=True, flash=True, number=True):
    rng = np.random.default_rng(1)
    bg = rng.integers(120, 170, (H, W, 3), dtype=np.uint8)
    for x, y in PLUGS:
        cv2.rectangle(bg, (x, y), (x + 14, y + 14), (20, 20, 20), -1)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(n):
        f = bg.copy()
        cv2.putText(f, f"12:00:{i // FPS:02d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)                       # burned-in clock
        if number:
            f[H - 24:H - 4, W - 24:W - 4] = i % 250           # frame number, for checks
        if bee and i in BEE_FRAMES:
            x = 480 + (i - BEE_FRAMES.start) * 2
            cv2.rectangle(f, (x, 120), (x + 12, 132), (15, 15, 15), -1)
        if flash and i in FLASH_FRAMES:
            f = np.clip(f.astype(int) + 90, 0, 255).astype(np.uint8)
        out.write(f)
    out.release()


def detect_dark(frames):
    """A detector that, like SAM 3 on a bee hotel, boxes every dark object —
    real bees and nest plugs alike."""
    out = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g[H - 30:, W - 30:] = 255                           # ignore the frame-number patch
        g[:20, :160] = 255                                  # and the clock
        n, _l, st, _c = cv2.connectedComponentsWithStats((g < 50).astype(np.uint8), 8)
        out.append([{"x": int(st[k][0]), "y": int(st[k][1]), "w": int(st[k][2]),
                     "h": int(st[k][3]), "class": "bee", "confidence": 0.6}
                    for k in range(1, n) if st[k][4] >= 20])
    return out


@pytest.fixture
def clip_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_picks_the_moving_bee_and_drops_the_plugs(clip_dir):
    p = os.path.join(clip_dir, "bee.avi")
    _write(p)
    res = sl.sample_label_clip(p, detect_dark, max_frames=5, candidates=10, min_gap_s=0.2)
    picks = res["picks"]
    assert picks, "the moving bee should be picked"
    assert all(n in BEE_FRAMES or n in range(BEE_FRAMES.start, BEE_FRAMES.stop + 5)
               for n in (p["n"] for p in picks))
    for p_ in picks:
        assert p_["boxes"], "each pick carries its detections"
        for b in p_["boxes"]:
            assert b["y"] < 200, "only the moving bee is kept — never a nest plug"
    gaps = [b - a for a, b in zip([p_["n"] for p_ in picks], [p_["n"] for p_ in picks][1:])]
    assert all(g >= round(0.2 * FPS) for g in gaps)


def test_a_clip_with_only_static_detections_gives_nothing(clip_dir):
    p = os.path.join(clip_dir, "empty.avi")
    _write(p, bee=False, flash=False)
    res = sl.sample_label_clip(p, detect_dark, max_frames=5)
    assert res["picks"] == []


def test_the_clock_is_not_motion(clip_dir):
    p = os.path.join(clip_dir, "empty.avi")
    _write(p, bee=False, flash=False, number=False)   # only the clock changes
    scan = sl.scan_clip(p, candidates=5)
    assert max(scan.scores) == 0


def test_a_frame_wide_change_is_handling_not_activity(clip_dir):
    p = os.path.join(clip_dir, "flash.avi")
    _write(p, bee=False, flash=True)
    scan = sl.scan_clip(p, candidates=5)
    assert all(scan.scores[i] == 0 for i in FLASH_FRAMES)


def test_picked_frame_numbers_match_sequential_decode(clip_dir):
    p = os.path.join(clip_dir, "bee.avi")
    _write(p)
    res = sl.sample_label_clip(p, detect_dark, max_frames=4, candidates=8, min_gap_s=0.2)
    cap = cv2.VideoCapture(p)
    decoded = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        decoded.append(f)
    for pick in res["picks"]:
        assert np.array_equal(pick["frame"], decoded[pick["n"]])
        assert abs(int(pick["frame"][H - 14, W - 14].mean()) - pick["n"] % 250) <= 6


def test_roi_excludes_motion_outside_it(clip_dir):
    p = os.path.join(clip_dir, "bee.avi")
    _write(p)
    scan = sl.scan_clip(p, candidates=5, roi=[0.0, 0.5, 0.6, 1.0])   # bee is top-right
    assert max(scan.scores) == 0


def test_the_saved_frame_is_not_masked(clip_dir):
    p = os.path.join(clip_dir, "bee.avi")
    _write(p)
    res = sl.sample_label_clip(p, detect_dark, max_frames=1, candidates=4)
    top_left = res["picks"][0]["frame"][:16, :120]
    assert top_left.max() > 200, "the clock stays in the stored image"


def test_motion_profile_shape():
    prof = sl.motion_profile([0] * 50 + [5] * 10 + [0] * 40, [55], buckets=10)
    assert prof["frames"] == 100 and prof["picked"] == [5] and max(prof["profile"]) == 100


def test_one_insect_keeps_one_box():
    """SAM 3 boxed a whole insect and three of its parts (first live run)."""
    whole = {"x": 322, "y": 419, "w": 162, "h": 161, "confidence": 0.575}
    parts = [{"x": 434, "y": 468, "w": 49, "h": 97, "confidence": 0.586},
             {"x": 335, "y": 419, "w": 101, "h": 73, "confidence": 0.577},
             {"x": 322, "y": 506, "w": 95, "h": 77, "confidence": 0.573}]
    other = {"x": 900, "y": 100, "w": 40, "h": 40, "confidence": 0.5}
    assert sl.drop_contained(parts + [whole, other]) == [whole, other]


def test_plugs_beside_a_moving_bee_are_not_kept():
    """The margin around motion is a quarter of a blob, not a whole one."""
    bee_blob = [(1150.0, 320.0, 1180.0, 342.0)]                  # 30 x 22 px of motion
    bee = {"x": 1152, "y": 319, "w": 32, "h": 23}
    plug_beside = {"x": 1237, "y": 323, "w": 27, "h": 32}         # ~57 px away
    assert sl.moving_detections([bee, plug_beside], bee_blob) == [bee]


def test_candidates_cover_the_whole_burst_not_its_start(clip_dir):
    """Top frames first, spacing after, kept only the start of a burst: a real
    2-minute clip moving in 2,268 frames gave 8 candidates of 15."""
    p = os.path.join(clip_dir, "bee.avi")
    _write(p, flash=False)
    scan = sl.scan_clip(p, candidates=6, spread=0, min_gap_s=0.2)
    ns = [c.n for c in scan.candidates]
    assert len(ns) == 6
    assert ns[-1] - ns[0] >= 25, ns
    assert all(b - a >= 5 for a, b in zip(ns, ns[1:]))


def test_a_box_counts_if_motion_touched_it_nearby(clip_dir):
    """A bee that pauses has no blob in that frame; motion within near_s counts."""
    p = os.path.join(clip_dir, "bee.avi")
    _write(p, flash=False)
    scan = sl.scan_clip(p, candidates=6, spread=0, min_gap_s=0.2)
    for c in scan.candidates:
        assert set(c.blobs) <= set(c.near)
        assert len(c.near) > len(c.blobs)
