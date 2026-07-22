"""Individual-bee identification (colour marks today, printed tags later).

The tracker has always had the hook — ``BeeTracking(identifier=...)`` calling
``identify(frame, bbox)`` per confirmed track — but nothing ever implemented a
decoder, so ``bee_id`` was always empty. This package fills it.

Two consumers, one contract:

* the tracker, per frame on the full-resolution frame (GPU worker);
* a web-side pass over the per-track crops already stored in S3, which needs no
  GPU and works on videos that were analysed before this existed.

See :mod:`beemonitor.identification.base` for the contract.
"""

from .base import BaseIdentifier, BeeIdentifierManager, Identification, crop_bbox
from .color_identifier import (
    EXTENDED_PALETTE,
    QUEEN_MARKING_PALETTE,
    ColorIdentifier,
    available_marker_types,
    build_identifier,
    hue_distance,
)
from .species import (
    IMAGE_SIZE,
    NON_BEE_TAXA,
    SpeciesIdentifier,
    SpeciesVote,
    taxa,
    taxon_ranks,
)

__all__ = [
    "BaseIdentifier",
    "BeeIdentifierManager",
    "ColorIdentifier",
    "Identification",
    "QUEEN_MARKING_PALETTE",
    "EXTENDED_PALETTE",
    "IMAGE_SIZE",
    "NON_BEE_TAXA",
    "SpeciesIdentifier",
    "SpeciesVote",
    "available_marker_types",
    "build_identifier",
    "crop_bbox",
    "hue_distance",
    "taxa",
    "taxon_ranks",
]
