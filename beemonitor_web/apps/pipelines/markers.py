"""Read individual bee markers from the per-track crops already in S3.

Every tracking job uploads a handful of crops per track plus a ``track_crops.csv``
index (``track_id, frame, crop_key``) — that has been happening since the "bee id
feature" commit, whose actual contribution was collecting this training data. The
decoder that was supposed to consume it never got written, so ``bee_id`` in the
tracking CSV is always empty.

This module closes that loop **without touching the GPU worker**: it pulls those
crops, runs :mod:`beemonitor.identification` over them on CPU, and votes per
track. Consequences worth knowing:

* it works on videos analysed *before* any of this existed — no re-run, no
  GPU spend;
* it needs no SageMaker image rebuild or endpoint rollover to ship;
* recall is bounded by what got saved (``crops_per_track``, default ~10) and the
  crops are re-compressed JPEGs, so it will read fewer marks than a decoder
  running on full frames inside the tracker would.

**Voting matters.** The tracker's own hook takes the first confident reading and
never revisits it (``Track.set_bee_id`` keeps the max-confidence answer, and the
call site is gated on ``bee_id is None``), so one bad decode there would brand a
bee for its whole trajectory. Here we decode every available crop and take the
majority, which is what makes a single misread survivable.
"""

import io
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Per-track crop cap. Crops are small (a few KB) but this is one S3 GET each, so
# keep the fan-out bounded on tracks that saved a lot.
MAX_CROPS_PER_TRACK = 12


def _load_crop_index(job_result):
    """Return {track_id: [crop_key, ...]} for a finished job.

    Prefers the ``crops_manifest`` already embedded in the job's summary stats
    (no S3 read at all); falls back to parsing ``track_crops.csv``.
    """
    result = job_result or {}
    summary = result.get("summary_stats") or {}
    manifest = summary.get("crops_manifest") or {}
    if manifest:
        return {str(k): list(v or []) for k, v in manifest.items()}

    path = result.get("crops_csv_path") or ""
    if not path:
        return {}
    from . import ops

    df = ops._read_csv(path)
    if df is None or len(df) == 0:
        return {}
    cols = {c.lower(): c for c in df.columns}
    track_col = cols.get("track_id") or cols.get("track")
    key_col = cols.get("crop_key") or cols.get("key") or cols.get("path")
    if not track_col or not key_col:
        return {}
    index = defaultdict(list)
    for _, row in df.iterrows():
        key = str(row[key_col] or "").strip()
        if key:
            index[str(row[track_col])].append(key)
    return dict(index)


def _fetch_crop(storage, key):
    """Download one crop and decode it to a BGR array, or None."""
    import cv2
    import numpy as np

    try:
        buf = io.BytesIO()
        storage.download_to_stream("processed", key, buf)
        data = buf.getvalue()
        if not data:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        logger.debug("could not read crop %s", key, exc_info=True)
        return None


def identify_from_crops(job_result, marker_type="auto", max_crops=MAX_CROPS_PER_TRACK):
    """Decode markers for every track that has crops.

    Returns the same shape ``ops.marker_identities`` produces, so the executor
    can hand either straight to the template, or ``None`` when there is nothing
    to work from (no crops indexed, or no decoder for this marker type).
    """
    from beemonitor.identification import build_identifier

    identifier = build_identifier(marker_type)
    if identifier is None:
        return None

    index = _load_crop_index(job_result)
    if not index:
        return None

    from config.storage import get_s3_client

    storage = get_s3_client()

    rows, markers = [], set()
    for track_id, keys in sorted(index.items()):
        votes = defaultdict(list)   # marker -> [confidence, ...]
        method_of = {}
        read = 0
        for key in list(keys)[:max_crops]:
            image = _fetch_crop(storage, key)
            if image is None:
                continue
            read += 1
            result = identifier.identify(image)
            if not result:
                continue
            marker, method, confidence = result
            votes[marker].append(float(confidence))
            method_of[marker] = method

        if not votes:
            continue
        # Majority wins; mean confidence breaks ties, so a marker read twice
        # weakly still beats one read once strongly only if it has more votes.
        marker = max(votes, key=lambda m: (len(votes[m]), sum(votes[m]) / len(votes[m])))
        confidences = votes[marker]
        rows.append({
            "track": _as_int(track_id),
            "marker": marker,
            "method": method_of.get(marker, "color"),
            "confidence": round(sum(confidences) / len(confidences), 3),
            "votes": len(confidences),
            "crops_read": read,
        })
        markers.add(marker)

    return {
        "identified_tracks": len(rows),
        "unique_markers": len(markers),
        "rows": rows,
        "source": "crops",
    }


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
