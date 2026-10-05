"""Species and marker identity for a track, by vote over every crop of it.

The tracker saves a padded crop of every frame each track was detected in
(``crops/<video>/track_NNNN/frame_MMMMMM.jpg``). A track is one animal, so after
tracking each of its crops is classified and casts one vote; the track takes the
label with the most votes, ties broken by mean confidence. Every crop votes its
top-1 — there is no confidence floor, because a floor on small, blurry crops is
exactly what left tracks with no species at all. How firm a call is shows in the
vote share and mean confidence instead.

Classifiers are duck-typed:

* species: ``classify_images(list of BGR arrays) -> list of (label, conf) | None``
* marker:  ``identify(BGR array) -> (marker, method, conf) | None``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2

from .species import SpeciesVote

logger = logging.getLogger(__name__)

_TRACK_RE = re.compile(r"track_(\d+)$")
_FRAME_RE = re.compile(r"frame_(\d+)$")

# Crops per forward pass. Crops are small; this bounds memory, not speed.
BATCH = 64


def track_crop_files(crops_root) -> dict:
    """``{track_id: [(frame, path), ...]}`` in frame order, from the tracker's
    crop folders (any depth under ``crops_root``)."""
    out: dict = {}
    root = Path(crops_root)
    if not root.exists():
        return out
    for path in root.glob("**/track_*/frame_*.jpg"):
        t = _TRACK_RE.search(path.parent.name)
        f = _FRAME_RE.search(path.stem)
        if not (t and f):
            continue
        out.setdefault(int(t.group(1)), []).append((int(f.group(1)), path))
    for crops in out.values():
        crops.sort()
    return out


def vote_tracks(crops_root, species=None, marker=None, batch: int = BATCH):
    """Classify every crop of every track and vote.

    Returns ``(tracks, rows)``: ``tracks`` is ``{track_id: {...}}`` with the
    species and/or marker result and its vote counts; ``rows`` is one dict per
    crop with what each model read, for ``track_votes.csv``.
    """
    files = track_crop_files(crops_root)
    tracks, rows = {}, []
    for track_id, crops in files.items():
        species_vote, marker_vote = SpeciesVote(), SpeciesVote()
        species_seen = marker_seen = 0
        for start in range(0, len(crops), batch):
            chunk = crops[start:start + batch]
            images = [cv2.imread(str(p)) for _f, p in chunk]
            readings = [None] * len(chunk)
            if species is not None:
                try:
                    readings = species.classify_images(images)
                except Exception:
                    logger.exception("species classification failed for track %s", track_id)
            for (frame, path), image, reading in zip(chunk, images, readings):
                row = {"track_id": track_id, "frame": frame, "crop": path.name}
                if species is not None and image is not None:
                    species_seen += 1
                    if reading:
                        label, conf = reading
                        species_vote.add(label, conf)
                        row.update(species=label, species_confidence=round(float(conf), 4))
                if marker is not None and image is not None:
                    marker_seen += 1
                    try:
                        read = marker.identify(image)
                    except Exception:
                        read = None
                    if read:
                        label, _method, conf = read
                        marker_vote.add(str(label), conf)
                        row.update(marker=str(label), marker_confidence=round(float(conf), 4))
                rows.append(row)

        result = {"crops": len(crops)}
        if species is not None:
            result.update(_summary("taxon", species_vote, species_seen))
        if marker is not None:
            result.update(_summary("marker", marker_vote, marker_seen))
        tracks[track_id] = result
    return tracks, rows


def _summary(prefix, vote, seen):
    """Winner, its mean confidence, votes, crops read and the winner's share."""
    win = vote.winner()
    if not win:
        return {prefix: None, f"{prefix}_confidence": None, f"{prefix}_votes": 0,
                f"{prefix}_crops": seen, f"{prefix}_vote_share": 0.0}
    label, confidence, votes = win
    return {prefix: label, f"{prefix}_confidence": confidence, f"{prefix}_votes": votes,
            f"{prefix}_crops": seen,
            f"{prefix}_vote_share": round(votes / max(1, vote.frames), 3)}
