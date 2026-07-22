"""BeeMachine species classification (Brian Spiesman, 2024-02-13).

EfficientNetV2-S over 354 bee taxa at 300x300, 93.8% validation accuracy. Unlike
the marker decoders in this package it answers *what species*, not *which
individual* — so its output lands in the tracking CSV's ``taxon`` column, which
already flows through events, interactions and the activity feed.

**Runs inside the tracking pass.** The worker has every decoded frame in memory,
so classifying there means every frame of a trajectory gets a vote at full
resolution, for no S3 cost. One blurred frame can't rename a bee: the winner is
decided by :class:`SpeciesVote` over the whole track, not by whichever frame
happened to come first.

Model format
------------
The published artefact is Keras ``.h5``. The GPU worker is a PyTorch image
(torch + ultralytics + transformers) and adding TensorFlow to it is ~600 MB plus
a CUDA-coexistence problem, so the intended deployment is a **one-time offline
conversion to ONNX**::

    pip install tf2onnx tensorflow
    python -m tf2onnx.convert \\
        --keras EfficientNetV2S_300_fp32_2_8_2024.h5 \\
        --output beemachine_v2s_300.onnx --opset 17

then upload the ``.onnx`` beside the YOLO weights in the models bucket so
``ModelManager`` fetches it like any other. The Keras path below still works if
TensorFlow happens to be present (handy for validating the conversion), but
onnxruntime is what production should use.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import BaseIdentifier, crop_bbox

logger = logging.getLogger(__name__)

# The model's native input size. Fixed by the architecture — do not tune.
IMAGE_SIZE = 300

_ASSETS = Path(__file__).parent / "assets"
_TAXA_FILE = _ASSETS / "beemachine_taxa.json"

# Categories that are not bees. Kept as real classes because the model was
# trained to recognise them — a confident "Syrphidae" is a useful answer, and
# far better than silently reporting the nearest bee.
NON_BEE_TAXA = frozenset({"Coleoptera", "Diptera", "Lepidoptera", "Syrphidae", "Wasp"})


@lru_cache(maxsize=1)
def taxa():
    """The ordered class list. Index i is the model's output unit i."""
    with _TAXA_FILE.open() as fh:
        return tuple(json.load(fh)["categories"])


def taxon_ranks(name: str) -> dict:
    """Split a BeeMachine label into ranks.

    Labels are ``Genus`` or ``Genus_species``; the non-bee classes are coarser
    groups (an order or a family) and get no genus.
    """
    if not name or name in NON_BEE_TAXA:
        return {}
    parts = name.split("_")
    ranks = {"genus": parts[0]}
    if len(parts) >= 2:
        ranks["species"] = f"{parts[0]} {' '.join(parts[1:])}"
    return ranks


class SpeciesIdentifier(BaseIdentifier):
    """Classify a cropped insect into one of the BeeMachine taxa.

    Satisfies the same ``identify(frame, bbox)`` contract as the marker decoders,
    so it can be injected wherever they can.

    Args:
        model_path: ``.onnx`` (preferred) or Keras ``.h5``.
        min_confidence: below this the reading is discarded. The model always
            emits a softmax over 354 classes, so *something* always wins — the
            floor is what stops a blurred wing from being reported as a species.
        session: a pre-built runtime, for tests.
    """

    method = "species"

    def __init__(self, model_path=None, min_confidence: float = 0.5, session=None,
                 min_crop_side: int = 24):
        self.model_path = str(model_path or os.environ.get("BEEMACHINE_MODEL_PATH", ""))
        self.min_confidence = float(min_confidence)
        self.min_crop_side = int(min_crop_side)
        self._session = session
        self._kind = "stub" if session is not None else None

    # ── runtime ──────────────────────────────────────────────────────────────
    def _runtime(self):
        """Lazily build the inference session (ONNX preferred, Keras fallback)."""
        if self._session is not None:
            return self._session
        if not self.model_path:
            raise RuntimeError(
                "No BeeMachine model configured — set BEEMACHINE_MODEL_PATH or "
                "pass model_path.")
        if self.model_path.endswith(".onnx"):
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = set(onnxruntime.get_available_providers())
            self._session = onnxruntime.InferenceSession(
                self.model_path, providers=[p for p in providers if p in available])
            self._kind = "onnx"
        else:
            from tensorflow import keras

            self._session = keras.models.load_model(self.model_path, compile=False)
            self._kind = "keras"
        logger.info("BeeMachine loaded (%s) from %s", self._kind, self.model_path)
        return self._session

    def preprocess(self, region):
        """Crop -> (1, 300, 300, 3) float32 batch, scaled to 0..1.

        Both steps were verified against the reference image the model ships
        with (a bumblebee on Echinacea), because getting either wrong fails
        *silently but confidently*:

        * **0..1, not 0..255.** This export has no internal rescaling layer, so
          feeding 0..255 drives the network into saturation — it returned
          ``Xylocopa_violacea`` at exactly 1.0000 confidence for that bumblebee.
          Scaled to 0..1 the same image gives ``Bombus_fervidus`` 0.911 with an
          all-Bombus top-3. High confidence is not evidence of correct
          preprocessing; a calibrated distribution is.
        * **BGR->RGB.** OpenCV decodes BGR, Keras trained on RGB. Skipping the
          swap also returns a confident wrong answer (``Megachile``).
        """
        import cv2

        if region is None or region.size == 0:
            return None
        if region.ndim != 3 or region.shape[2] != 3:
            return None
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32)[None, ...] / 255.0

    def _infer(self, batch):
        """Run the batch, dispatching on what the session actually offers.

        Duck-typed rather than keyed off the loader, so an injected test double
        or a future runtime works as long as it looks like one of the two.
        """
        session = self._runtime()
        if hasattr(session, "run") and hasattr(session, "get_inputs"):
            name = session.get_inputs()[0].name
            return np.asarray(session.run(None, {name: batch})[0])
        if hasattr(session, "predict"):
            return np.asarray(session.predict(batch, verbose=0))
        raise TypeError(f"unusable inference session: {type(session).__name__}")

    def _decode(self, probs):
        """One row of model output -> a reading, or None."""
        names = taxa()
        probs = np.asarray(probs).reshape(-1)
        if probs.size != len(names):
            logger.warning(
                "BeeMachine output has %d units but %d taxa are configured — "
                "refusing to guess a mapping", probs.size, len(names))
            return None
        # Softmax if the export left logits unnormalised.
        if probs.min() < 0 or not (0.99 <= float(probs.sum()) <= 1.01):
            shifted = probs - probs.max()
            exp = np.exp(shifted)
            probs = exp / exp.sum()
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        if confidence < self.min_confidence:
            return None
        return (names[index], self.method, round(confidence, 4))

    def _usable(self, region):
        """Reject crops too small to classify.

        A handful of pixels upscaled to 300x300 is noise, and a low-quality vote
        is worse than no vote — it is exactly the input that produces confident
        nonsense, and majority voting only helps when most votes are informative.
        """
        if region is None or region.size == 0:
            return False
        if region.ndim != 3 or region.shape[2] != 3:
            return False
        h, w = region.shape[:2]
        return min(h, w) >= self.min_crop_side

    def identify(self, frame, bbox=None):
        """Return ``(taxon, "species", confidence)`` or None."""
        try:
            region = crop_bbox(frame, bbox)
            if not self._usable(region):
                return None
            batch = self.preprocess(region)
            if batch is None:
                return None
            scores = self._infer(batch)
            if scores is None or getattr(scores, "size", 0) == 0:
                return None
            return self._decode(np.asarray(scores)[0])
        except Exception:  # inside the per-frame loop — never propagate
            logger.debug("BeeMachine classification failed", exc_info=True)
            return None

    def identify_batch(self, frame, bboxes):
        """Classify several crops of one frame in a single forward pass.

        Returns a list aligned with ``bboxes`` (None where a crop was unusable or
        below the confidence floor). Batching matters because at batch 1 the GPU
        is latency-bound — it spends most of its time on launch overhead rather
        than arithmetic — so the tracker classifies all of a frame's tracks
        together instead of one call each.
        """
        results = [None] * len(bboxes)
        usable = []
        try:
            for i, bbox in enumerate(bboxes):
                region = crop_bbox(frame, bbox)
                if not self._usable(region):
                    continue
                batch = self.preprocess(region)
                if batch is not None:
                    usable.append((i, batch[0]))
            if not usable:
                return results
            stacked = np.stack([b for _i, b in usable]).astype(np.float32)
            scores = np.asarray(self._infer(stacked))
            for row, (i, _b) in enumerate(usable):
                results[i] = self._decode(scores[row])
        except Exception:
            logger.debug("BeeMachine batch classification failed", exc_info=True)
        return results


class SpeciesVote:
    """Accumulate per-frame species readings for one trajectory.

    A track is one animal, so its frames should agree; where they don't, the
    majority is a far better estimate than any single frame. Ties break on summed
    confidence, so a class seen twice weakly loses to one seen twice strongly.
    """

    __slots__ = ("counts", "weights", "frames")

    def __init__(self):
        self.counts = {}
        self.weights = {}
        self.frames = 0

    def add(self, taxon: str, confidence: float = 1.0) -> None:
        if not taxon:
            return
        self.counts[taxon] = self.counts.get(taxon, 0) + 1
        self.weights[taxon] = self.weights.get(taxon, 0.0) + float(confidence)
        self.frames += 1

    def winner(self):
        """``(taxon, mean_confidence, votes)``, or None if nothing was read."""
        if not self.counts:
            return None
        taxon = max(self.counts, key=lambda t: (self.counts[t], self.weights[t]))
        votes = self.counts[taxon]
        return taxon, round(self.weights[taxon] / votes, 4), votes

    def as_dict(self):
        result = self.winner()
        if not result:
            return {"taxon": None, "taxon_confidence": 0.0, "taxon_votes": 0,
                    "taxon_frames": self.frames}
        taxon, confidence, votes = result
        return {"taxon": taxon, "taxon_confidence": confidence,
                "taxon_votes": votes, "taxon_frames": self.frames,
                "taxon_agreement": round(votes / max(1, self.frames), 3)}
