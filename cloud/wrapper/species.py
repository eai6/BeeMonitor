"""Per-track species identification with BioCLIP.

Runs inside the GPU tracking job (the only place the tracker's per-track crop
images exist — they're written to local disk and never uploaded). After
tracking, for each ``track_NNNN`` crop directory we send a few crops to the
BioCLIP SageMaker endpoint (the same sync contract the web monitor app uses),
vote a top-1 species per track, and aggregate an observations summary.

Gated by the caller on ``run_species`` + a configured endpoint; failures are
non-fatal (species is an enrichment, not the core tracking result).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Crops to classify per track (they're near-duplicates; a few is enough to vote
# and keeps BioCLIP invocations — hence cost + latency — bounded).
CROPS_PER_TRACK = 3
MAX_TRACKS = 400  # safety cap on total classification work per video


def _runtime(region: str):
    import boto3
    from botocore.config import Config
    return boto3.client(
        "sagemaker-runtime", region_name=region,
        # BioCLIP is serverless/scale-to-zero — allow for cold starts.
        config=Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 2}),
    )


def _classify_crop(rt, endpoint: str, jpeg: bytes, candidate_taxa=None) -> dict | None:
    """One crop -> top prediction dict {score, common_name, ranks} or None."""
    body = {"image_b64": base64.b64encode(jpeg).decode("ascii")}
    if candidate_taxa:
        body["candidate_taxa"] = list(candidate_taxa)
        body["rank"] = "species"
    try:
        resp = rt.invoke_endpoint(
            EndpointName=endpoint, ContentType="application/json",
            Accept="application/json", Body=json.dumps(body).encode("utf-8"),
        )
        data = json.loads(resp["Body"].read().decode("utf-8"))
        preds = data.get("predictions", []) if isinstance(data, dict) else data
        return preds[0] if preds else None
    except Exception as e:  # noqa: BLE001
        logger.warning("bioclip invoke failed: %s", e)
        return None


def _track_dirs(output_dir: str):
    """Yield (track_id, [crop_paths]) for each track_NNNN dir under output_dir."""
    for root, dirs, _files in os.walk(output_dir):
        for d in dirs:
            if not d.startswith("track_"):
                continue
            try:
                track_id = int(d.split("_")[1])
            except (IndexError, ValueError):
                continue
            tdir = os.path.join(root, d)
            crops = sorted(
                os.path.join(tdir, f) for f in os.listdir(tdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if crops:
                yield track_id, crops


def classify_tracks(output_dir: str, endpoint: str, region: str,
                    candidate_taxa=None, min_confidence: float = 0.0) -> dict:
    """Classify every track's crops. Returns
    {"per_track": {tid: {species, common_name, confidence, votes}},
     "observations": [{species, common_name, track_count, avg_confidence}, ...],
     "tracks_classified": int}.
    """
    rt = _runtime(region)
    per_track: dict[int, dict] = {}
    tracks = list(_track_dirs(output_dir))
    if len(tracks) > MAX_TRACKS:
        logger.info("species: capping %d tracks to %d", len(tracks), MAX_TRACKS)
        tracks = tracks[:MAX_TRACKS]

    for track_id, crops in tracks:
        votes = Counter()
        scores = defaultdict(list)
        commons = {}
        for path in crops[:CROPS_PER_TRACK]:
            try:
                with open(path, "rb") as fh:
                    jpeg = fh.read()
            except OSError:
                continue
            pred = _classify_crop(rt, endpoint, jpeg, candidate_taxa)
            if not pred:
                continue
            ranks = pred.get("ranks") or {}
            species = ranks.get("species") or pred.get("common_name") or "unknown"
            if not species or species == "unknown":
                continue
            votes[species] += 1
            scores[species].append(float(pred.get("score", 0.0)))
            commons[species] = pred.get("common_name") or commons.get(species, "")
        if not votes:
            continue
        top_species, _ = votes.most_common(1)[0]
        conf = round(sum(scores[top_species]) / len(scores[top_species]), 4)
        if conf < min_confidence:
            continue
        per_track[track_id] = {
            "species": top_species,
            "common_name": commons.get(top_species, ""),
            "confidence": conf,
            "votes": votes[top_species],
        }

    # Aggregate observations: one row per species, counting tracks.
    obs_counter = Counter()
    obs_conf = defaultdict(list)
    obs_common = {}
    for rec in per_track.values():
        sp = rec["species"]
        obs_counter[sp] += 1
        obs_conf[sp].append(rec["confidence"])
        obs_common[sp] = rec["common_name"]
    observations = [
        {"species": sp, "common_name": obs_common.get(sp, ""),
         "track_count": n,
         "avg_confidence": round(sum(obs_conf[sp]) / len(obs_conf[sp]), 4)}
        for sp, n in obs_counter.most_common()
    ]

    logger.info("species: classified %d/%d tracks into %d taxa",
                len(per_track), len(tracks), len(observations))
    return {
        "per_track": per_track,
        "observations": observations,
        "tracks_classified": len(per_track),
    }


def write_track_species_csv(per_track: dict, path: str) -> None:
    """Write a track_id -> species CSV for download."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "species", "common_name", "confidence", "votes"])
        for tid in sorted(per_track):
            r = per_track[tid]
            w.writerow([tid, r["species"], r["common_name"], r["confidence"], r["votes"]])
