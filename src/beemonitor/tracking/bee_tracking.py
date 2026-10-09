"""
BeeTracking - v2.4 YOLO-Only with ADAPTIVE Tracker + Bee Identification
========================================================================

Simplified bee tracking using YOLO-only detection with adaptive, resolution and FPS-independent tracking.

v2.4 Changes:
- Added bee identification support (color, number, QR code)
- taxon field from YOLO class label (always present)
- bee_id field for individual identification (optional)
- Configurable identifier via ColorIdentifier or BeeIdentifierManager

v2.3 Changes:
- Pass frame_height to tracker for resolution-relative fallback
- Robust bee size with IQR outlier rejection (in BeeTracker)
- Track IDs start from 1 for confirmed tracks only

v2.2 Changes:
- YOLO-only detection (no FGBG_YOLO mode)
- Two-mode optimization (motion detection + YOLO tracking)
- Adaptive tracker with FPS and bee-size relative thresholds
- Auto-calculates bee size from detections
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from ultralytics import YOLO

import os
import queue
import threading

from beemonitor import cancellation
from beemonitor.core.profiling import PROFILER
from beemonitor.detection.yolo_detector import YOLODetector
from beemonitor.detection.blob_detector import BlobDetector
from beemonitor.tracking.mot.bee_tracker import BeeTracker

logger = logging.getLogger(__name__)


# How many frames the reader thread may run ahead of the tracker.
#
# Decode is CPU and inference is GPU; run serially they alternate and neither
# saturates — a measured run showed one of four vCPUs pinned with the GPU
# memory at 28%. A bounded queue overlaps them. Bounded, because a 1080p BGR
# frame is ~6 MB: the default depth costs ~150 MB, unbounded costs the video.
#
# 0 disables the thread and decodes inline. That is the original path, kept
# because it is what the equivalence test compares the threaded one against.
FRAME_QUEUE_DEPTH = int(os.environ.get("BEEMONITOR_FRAME_QUEUE", "24"))

# Sentinel posted by the reader when there are no more frames. A distinct object
# rather than None so it can never be confused with a decode result.
_END_OF_FRAMES = object()


# The one shape the tracker consumes. BeeTracker.update() and
# _build_detections_df both read it POSITIONALLY, so the layout is load-bearing
# and was previously hand-built at three call sites with the format recorded
# only in a passing comment. One converter, named, so a fourth caller cannot
# quietly disagree with it.
TRACKER_ROW_FIELDS = ("x1", "y1", "x2", "y2", "confidence", "source", "taxon")


def to_tracker_rows(detections, source="yolo"):
    """``Detection`` objects -> the positional rows BeeTracker.update expects."""
    return [list(det.bbox) + [det.confidence, source, det.label]
            for det in detections]


def _decode_into(cap, start_frame, end_frame, frame_queue, stop_event):
    """Decode frames into ``frame_queue`` until EOF or ``end_frame``.

    Runs on the reader thread. Every exit path — clean, stopped, or raised —
    posts exactly one sentinel, so the consumer can never block forever. Puts
    use a timeout rather than blocking outright so a consumer that has gone away
    cannot wedge this thread against a full queue.
    """
    frame_num = start_frame
    try:
        while not stop_event.is_set():
            if end_frame is not None and frame_num >= end_frame:
                break
            with PROFILER.stage("decode"):
                ret, frame = cap.read()
            if not ret:
                break
            while not stop_event.is_set():
                try:
                    frame_queue.put((frame_num, frame), timeout=0.5)
                    break
                except queue.Full:
                    continue
            frame_num += 1
    except Exception:
        logger.exception("Frame reader failed at frame %s", frame_num)
    finally:
        while True:
            try:
                frame_queue.put(_END_OF_FRAMES, timeout=0.5)
                break
            except queue.Full:
                if stop_event.is_set():
                    break


# Crops are the record a species or marker model reads later, so keep them close
# to the decoded pixels.
CROP_JPEG_QUALITY = 95


def crop_sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian of the crop's greyscale — high for a crisp
    crop, low for a motion-blurred or out-of-focus one. Only compared within a
    track, so its scale doesn't matter."""
    if crop is None or crop.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def padded_box(bbox, width, height, padding=0.25, min_padding_px=16):
    """``bbox`` grown by ``padding`` × its size on each side (at least
    ``min_padding_px``), clamped to the frame, as ints — or None if empty."""
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    pad_x = max(min_padding_px, (x2 - x1) * padding)
    pad_y = max(min_padding_px, (y2 - y1) * padding)
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(int(width), int(round(x2 + pad_x)))
    y2 = min(int(height), int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iter_frames(cap, start_frame, end_frame):
    """``(frame_num, frame)`` in order, decoded ahead on a reader thread.

    Order is exactly the serial loop's: one frame at a time, none skipped. Only
    *when* the decode happens changes, so process_frame, the tracker and the
    MOG2 background model — all of which are stateful and order-dependent —
    cannot tell the difference.
    """
    if FRAME_QUEUE_DEPTH <= 0:
        frame_num = start_frame
        while end_frame is None or frame_num < end_frame:
            with PROFILER.stage("decode"):
                ret, frame = cap.read()
            if not ret:
                return
            yield frame_num, frame
            frame_num += 1
        return

    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_DEPTH)
    stop_event = threading.Event()
    reader = threading.Thread(
        target=_decode_into,
        args=(cap, start_frame, end_frame, frame_queue, stop_event),
        name="beemonitor-frame-reader",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            item = frame_queue.get()
            if item is _END_OF_FRAMES:
                return
            yield item
    finally:
        # Covers the consumer raising, or a caller abandoning the generator:
        # tell the reader to stop, then drain so it is never blocked on a put.
        stop_event.set()
        while reader.is_alive():
            try:
                frame_queue.get(timeout=0.1)
            except queue.Empty:
                pass
        reader.join(timeout=5.0)


class BeeTracking:
    """
    YOLO-only bee tracking with two-mode optimization and adaptive parameters.
    
    Features:
    - Motion detection mode (fast blob detection)
    - Tracking mode (YOLO when motion detected)
    - Adaptive tracker (resolution & FPS independent)
    - Auto-calculates bee size
    - Configurable distance multipliers for tuning
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        confidence_threshold: float = 0.25,
        roi: Optional[Tuple[int, int, int, int]] = None,
        # The ROI traced as a polygon (pixel coords), when the user drew one
        # instead of dragging a box. The crop stays `roi` (its bounding box);
        # pixels outside the outline are blanked before detection, so grass, sky
        # or a neighbouring trap inside that box produce nothing to track.
        roi_polygon: Optional[list] = None,
        # Adaptive tracker parameters
        max_age_seconds: float = 1.0,
        min_hits_seconds: float = 0.1,
        max_resurrection_seconds: float = 0.5,
        match_distance_multiplier: float = 8.0,
        resurrection_search_multiplier: float = 3.0,
        duplicate_distance_multiplier: float = 1.2,
        iou_threshold: float = 0.3,
        # Bee identification
        identifier=None,  # BeeIdentifierManager or ColorIdentifier instance
        # Species classifier (BeeMachine). Where `identifier` answers "which
        # individual", this answers "which species" — and votes across every
        # frame of a trajectory instead of sticking on the first read.
        species_classifier=None,
        # Votes to collect per track before a species is considered settled.
        # Majority voting saturates fast: at 70% per-frame accuracy, 25 votes is
        # 98.2% and 900 votes is 100% — 1.8 points for 36x the GPU time. Capping
        # is the difference between ~$0.02 and ~$0.0002 of classification per
        # video. 0 = no cap (classify every frame).
        species_max_votes: int = 25,
        # Crop saving for identification training
        save_crops: bool = False,
        crop_output_dir: Optional[str] = None,
        crops_per_track: int = 0,
        crops_keep_sharpest: int = 0,
        # Margin added around each saved crop, as a fraction of the box's own
        # width/height per side, with a floor in pixels. The detector's box is
        # tight on the body, so an unpadded crop clips legs, antennae and wing
        # edges — the details a small bee is identified by.
        crop_padding: float = 0.25,
        crop_min_padding_px: int = 16,
        # Association algorithm (memory/43): "beetrack" (BeeTracker, default) or
        # one of mot.external.TRACKERS, with that tracker's own settings.
        tracker_kind: str = "beetrack",
        tracker_options: Optional[Dict[str, Any]] = None,
        # Pluggable detector: pass an injected BaseDetector (e.g. Sam3Detector)
        # to replace YOLO; default None builds the YOLO detector below.
        detector=None,
    ):
        """
        Initialize BeeTracking with YOLO-only detection and adaptive tracking.
        
        Args:
            yolo_model_path: Path to YOLO model weights
            confidence_threshold: YOLO confidence threshold
            roi: Region of interest (x1, y1, x2, y2)
            
            Adaptive tracker parameters (resolution & FPS independent):
                max_age_seconds: Max time without detection before track dies
                min_hits_seconds: Min time before track is confirmed
                max_resurrection_seconds: Max time to resurrect dead tracks
                match_distance_multiplier: Max distance for matching (multiplier of bee size)
                resurrection_search_multiplier: Search radius (multiplier of bee size)
                duplicate_distance_multiplier: Duplicate threshold (multiplier of bee size)
                iou_threshold: IoU threshold for matching
            
            Bee identification:
                identifier: Identifier instance (ColorIdentifier or BeeIdentifierManager)
            
            Crop saving (for identification training data):
                save_crops: Whether to save bbox crops of tracked bees
                crop_output_dir: Directory to save crops (default: ./crops)
                crops_per_track: Max crops per track; 0 (default) = every frame
                    the track was detected in
                crops_keep_sharpest: Keep only the N sharpest crops of each
                    track, replacing the blurriest as sharper ones arrive;
                    0 (default) = keep every crop
                crop_padding / crop_min_padding_px: margin around each crop
        """
        logger.info("Initializing BeeTracking (v2.4 YOLO-only with adaptive tracker + identification)")
        
        # YOLO detector
        logger.info(f"Loading YOLO model from {yolo_model_path}")

        # self.yolo_detector = YOLODetector(
        #     model_path=yolo_model_path,
        #     confidence_threshold=confidence_threshold
        # )

        # Injected detector wins (e.g. SAM 3 text-prompt); else build YOLO.
        # The attribute stays `yolo_detector` for minimal churn — it's just the
        # active detector, called at the three .detect() sites below.
        if detector is not None:
            logger.info(f"Using injected detector: {detector.get_source_name()}")
            self.yolo_detector = detector
        else:
            yolo_model = YOLO(yolo_model_path)
            self.yolo_detector = YOLODetector(
                model=yolo_model,
                conf_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        
        # Blob detector (for motion detection mode)
        logger.info("Initializing blob detector for motion detection")
        self.blob_detector = BlobDetector()
        
        # ROI
        self.roi = roi
        self.roi_polygon = list(roi_polygon) if roi_polygon and len(roi_polygon) >= 3 else None
        self._roi_mask = None          # rasterised polygon, cached per crop size
        self._roi_mask_shape = None
        if self.roi_polygon:
            logger.info(f"ROI polygon active ({len(self.roi_polygon)} corners) — "
                        "masking background inside the ROI box")

        # Video properties (will be set when video is opened)
        self.fps = None
        self.video_width = None
        self.video_height = None
        
        # Store tracker parameters for initialization
        self.tracker_params = {
            'max_age_seconds': max_age_seconds,
            'min_hits_seconds': min_hits_seconds,
            'max_resurrection_seconds': max_resurrection_seconds,
            'match_distance_multiplier': match_distance_multiplier,
            'resurrection_search_multiplier': resurrection_search_multiplier,
            'duplicate_distance_multiplier': duplicate_distance_multiplier,
            'iou_threshold': iou_threshold
        }
        
        # Tracker (will be initialized when video properties are known)
        self.tracker = None
        self.tracker_kind = (tracker_kind or "beetrack").lower()
        self.tracker_options = dict(tracker_options or {})
        
        # Bee identifier (optional)
        self.identifier = identifier

        # Species classification, voted per track (see SpeciesVote).
        self.species_classifier = species_classifier
        self.species_max_votes = int(species_max_votes)
        self.species_votes = {}   # track_id -> SpeciesVote

        # Raw pre-association detections from the last process_video call.
        # Populated there; None until a video has been processed.
        self.detections_df = None
        
        # Crop saving for identification training
        self.save_crops = save_crops
        self.crop_output_dir = crop_output_dir or './crops'
        self.crops_per_track = crops_per_track
        self.crops_keep_sharpest = max(0, int(crops_keep_sharpest or 0))
        # track_id -> min-heap of (sharpness, frame, path) of the crops on disk.
        self._crop_heaps = {}
        # path -> sharpness of every crop on disk, written beside the crops
        # (sharpness.csv) so the uploader never has to read them back.
        self._crop_sharpness = {}
        self.crop_padding = max(0.0, float(crop_padding))
        self.crop_min_padding_px = max(0, int(crop_min_padding_px))
        self.track_crop_counts = {}  # track_id -> number of crops saved
        
        # Two-mode system state
        self.enable_two_mode = True  # Can be set to False via config to force YOLO every frame
        self.mode = 'motion_detection'  # 'motion_detection' or 'tracking'
        self.frames_since_motion = 0
        self.motion_cooldown = 30  # Frames to stay in tracking mode after motion stops
        
        # Lookback buffer for catching motion before detection
        self.lookback_seconds = 0.5  # How far back to look when motion detected
        self.frame_buffer = []  # Ring buffer of (frame_num, frame) tuples
        self.lookback_frames = 0  # Will be set based on FPS
        self.processing_lookback = False  # Flag to prevent infinite loops
        
        id_status = "enabled" if identifier else "disabled"
        crop_status = f"enabled ({crops_per_track}/track)" if save_crops else "disabled"
        logger.info(f"BeeTracking initialized (v2.4, identification={id_status}, crops={crop_status})")
    
    def _initialize_tracker(self, fps: float, frame_height: int = None):
        """
        Initialize adaptive tracker with video FPS and frame height.
        
        Args:
            fps: Video frame rate
            frame_height: Video frame height for resolution-relative fallback
        """
        if self.tracker_kind != "beetrack":
            from beemonitor.tracking.mot.external import ExternalTracker
            self.tracker = ExternalTracker(
                self.tracker_kind, self.tracker_options, fps=fps,
                frame_size=(getattr(self, "video_width", 0) or 0, frame_height or 0))
        else:
            logger.info(f"Initializing adaptive tracker with FPS={fps}, frame_height={frame_height}")
            self.tracker = BeeTracker(
                fps=fps,
                bee_size=None,  # Auto-calculate from detections
                frame_height=frame_height,  # For resolution-relative fallback
                **self.tracker_params
            )
        
        # Set lookback buffer size based on FPS
        self.lookback_frames = int(fps * self.lookback_seconds)
        self.frame_buffer = []  # Reset buffer
        logger.info(f"Lookback buffer: {self.lookback_frames} frames ({self.lookback_seconds}s)")
        
        logger.info("Adaptive tracker initialized")
    
    def _get_video_properties(self, video_path: str) -> Tuple[float, int, int]:
        """
        Get video properties (FPS, width, height).
        
        Args:
            video_path: Path to video file
            
        Returns:
            (fps, width, height)
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Default to 30fps if unable to read
        if fps <= 0:
            logger.warning(f"Could not read FPS from video, defaulting to 30.0")
            fps = 30.0
        
        logger.info(f"Video properties: {width}x{height} @ {fps} fps")
        
        return fps, width, height
    
    def initialize_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Initialize tracking for a specific video.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to output folder
        """
        import os
        
        logger.info(f"Initializing video tracking for: {video_path}")
        if output_path:
            logger.info(f"Output path: {output_path}")
        
        # Get video properties
        self.fps, self.video_width, self.video_height = self._get_video_properties(video_path)
        logger.info(f"Video properties: {self.video_width}x{self.video_height} @ {self.fps} fps")
        
        # Initialize adaptive tracker with video FPS and frame height
        self._initialize_tracker(self.fps, self.video_height)
        
        # Reset crop counts for new video
        self.track_crop_counts = {}
        self._crop_heaps = {}
        self._crop_sharpness = {}
        
        # Extract video name for per-video crop folders
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Setup crop output directory (per-video to avoid track ID collisions)
        if self.save_crops:
            if output_path:
                # output_path might be a file (video) or directory
                # Use parent directory if it's a file
                if os.path.splitext(output_path)[1]:  # Has extension = file
                    output_dir = os.path.dirname(output_path)
                else:
                    output_dir = output_path
                # Include video name to separate crops from different videos
                self.crop_output_dir = os.path.join(output_dir, 'crops', video_name)
            else:
                # No output_path provided - use default with video name
                self.crop_output_dir = os.path.join('./crops', video_name)
            os.makedirs(self.crop_output_dir, exist_ok=True)
            abs_crop_dir = os.path.abspath(self.crop_output_dir)
            logger.info(f"Crop saving enabled: {abs_crop_dir} ({self.crops_per_track} per track)")
        
        # Initialize blob detector background
        # Note: BlobDetector builds background model internally from detect() calls
        # We don't need to manually feed it frames
        logger.info("Blob detector initialized (will build background on first detect() call)")
        
        # Optionally save initial background if output_path is provided
        # This will be saved after the first few frames are processed
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            self.background_save_path = os.path.join(output_path, "background.png")
            logger.info(f"Background will be saved to: {self.background_save_path}")
        else:
            self.background_save_path = None
    
    def _mask_roi(self, roi_frame: np.ndarray) -> np.ndarray:
        """Blank the parts of the ROI crop that fall outside the traced outline.

        A no-op unless the ROI was drawn as a polygon. The mask is rasterised
        once per crop size and reused; points are shifted by the crop origin so
        they line up whether or not the frame was cropped. Blanked pixels are
        constant, so the blob detector sees no motion there at all.
        """
        if not self.roi_polygon or roi_frame is None or roi_frame.size == 0:
            return roi_frame
        shape = roi_frame.shape[:2]
        if self._roi_mask is None or self._roi_mask_shape != shape:
            ox, oy = (self.roi[0], self.roi[1]) if self.roi else (0, 0)
            pts = np.array([[int(x - ox), int(y - oy)] for x, y in self.roi_polygon],
                           dtype=np.int32)
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            if not mask.any():   # outline doesn't overlap the crop — ignore it
                logger.warning("ROI polygon does not overlap the ROI crop — ignoring it")
                self.roi_polygon = None
                return roi_frame
            self._roi_mask, self._roi_mask_shape = mask, shape
        return cv2.bitwise_and(roi_frame, roi_frame, mask=self._roi_mask)

    def detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect if there is motion in the frame.

        Args:
            frame: Input frame
            
        Returns:
            True if motion detected, False otherwise
        """
        # Get blob detections (fast)
        blob_detections = self.blob_detector.detect(frame)
        
        # Motion detected if we have any blobs
        return len(blob_detections) > 0
    
    def _classify_species(self, frame: np.ndarray, tracks: List[Dict]):
        """Add a species vote per still-unsettled track, in one forward pass.

        Two things keep this cheap. Tracks that already have `species_max_votes`
        are skipped — voting accuracy saturates long before a full trajectory is
        consumed. The rest go through the classifier together, because at batch 1
        the GPU is latency-bound and spends most of its time on launch overhead.
        """
        from beemonitor.identification.species import SpeciesVote

        pending, boxes = [], []
        for track in tracks:
            track_id = track.get('track_id')
            if track_id is None:
                continue
            if self.species_max_votes > 0:
                vote = self.species_votes.get(track_id)
                if vote is not None and vote.frames >= self.species_max_votes:
                    continue
            bbox = (track.get('x1'), track.get('y1'),
                    track.get('x2'), track.get('y2'))
            if any(v is None for v in bbox):
                continue
            pending.append(track_id)
            boxes.append(bbox)

        if not pending:
            return
        for track_id, result in zip(
                pending, self.species_classifier.identify_batch(frame, boxes)):
            if not result:
                continue
            taxon, _method, confidence = result
            self.species_votes.setdefault(track_id, SpeciesVote()).add(taxon, confidence)

    def species_for(self, track_id):
        """The winning taxon for a track, or None while it has no votes."""
        vote = self.species_votes.get(track_id)
        return vote.winner() if vote else None

    def _give_frame(self, frame):
        """Trackers that look at the image (BoT-SORT's camera-motion
        compensation) get the frame; BeeTracker doesn't take one."""
        if hasattr(self.tracker, "frame"):
            self.tracker.frame = frame

    def _save_track_crops(self, frame: np.ndarray, frame_num: int):
        """Save a padded crop of every track detected in this frame.

        Every frame a track was *detected* in gets a crop, from its first
        frame. A frame where the tracker only predicted the bee (no detection,
        ``time_since_update > 0``) gets none: the stored box is from an earlier
        frame and the bee has moved, so the crop would be of the wrong place.

        A track has no id until it is confirmed (``min_hits``), so the crops of
        its tentative frames wait on the track object and are written under its
        id once it is confirmed. A track that dies tentative was noise; its
        crops go with it.
        """
        import os

        if not self.save_crops or self.tracker is None:
            return

        h, w = frame.shape[:2]
        for track in self.tracker.tracks:
            if track.time_since_update != 0:
                continue  # predicted, not detected, this frame

            box = padded_box(track.last_bbox, w, h,
                             self.crop_padding, self.crop_min_padding_px)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pending = getattr(track, "_pending_crops", None)
            if not track.is_confirmed:
                if pending is None:
                    pending = track._pending_crops = []
                pending.append((frame_num, crop.copy()))
                continue

            if pending:
                for pending_frame, pending_crop in pending:
                    self._write_crop(track.id, pending_frame, pending_crop)
                track._pending_crops = []
            self._write_crop(track.id, frame_num, crop)

    def _write_crop(self, track_id: int, frame_num: int, crop: np.ndarray):
        """Write one crop to crops/track_{id}/frame_{num}.jpg, honouring
        ``crops_per_track`` (0 = no cap) and ``crops_keep_sharpest``.

        With ``crops_keep_sharpest`` the crop is scored before it is encoded: a
        track already holding N sharper ones never writes it, and a sharper one
        replaces the track's blurriest on disk. A clip saved ~90k crops a track
        page shows 12 of; JPEG-encoding and uploading the rest was most of the
        job's time.
        """
        import heapq
        import os

        saved = self.track_crop_counts.get(track_id, 0)
        if self.crops_per_track > 0 and saved >= self.crops_per_track:
            return
        sharpness = crop_sharpness(crop)
        keep = self.crops_keep_sharpest
        heap = self._crop_heaps.setdefault(track_id, [])
        if keep and len(heap) >= keep and sharpness <= heap[0][0]:
            return
        track_dir = os.path.join(self.crop_output_dir, f'track_{track_id:04d}')
        os.makedirs(track_dir, exist_ok=True)
        crop_path = os.path.join(track_dir, f'frame_{frame_num:06d}.jpg')
        if not cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), CROP_JPEG_QUALITY]):
            logger.warning(f"Failed to save crop: {crop_path}")
            return
        self._crop_sharpness[crop_path] = sharpness
        if keep:
            if len(heap) >= keep:
                _s, _f, dropped = heapq.heapreplace(heap, (sharpness, frame_num, crop_path))
                self._crop_sharpness.pop(dropped, None)
                try:
                    os.remove(dropped)
                except OSError:
                    pass
                return  # one in, one out: the count is unchanged
            heapq.heappush(heap, (sharpness, frame_num, crop_path))
        self.track_crop_counts[track_id] = saved + 1

    def _write_crop_sharpness(self):
        """sharpness.csv beside the crops: relative path, sharpness."""
        import csv
        import os

        if not self._crop_sharpness:
            return
        with open(os.path.join(self.crop_output_dir, "sharpness.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["crop", "sharpness"])
            for path, s in sorted(self._crop_sharpness.items()):
                w.writerow([os.path.relpath(path, self.crop_output_dir), round(s, 2)])

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single frame with two-mode optimization.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with detections, tracks, and mode info
        """
        # Apply ROI for motion detection only
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
        else:
            roi_frame = frame
        # ...and, for a traced ROI, blank everything outside the outline so the
        # background the box still contains can't wake the tracker. Detection
        # itself stays full-frame (a bee is tracked past the ROI, as before).
        roi_frame = self._mask_roi(roi_frame)

        detections = []
        lookback_results = []  # Results from lookback frames
        
        # If two-mode disabled, force YOLO on every frame
        if not self.enable_two_mode:
            self.mode = "tracking"

        # TWO-MODE SYSTEM
        if self.mode == 'motion_detection':
            # Add frame to lookback buffer (only in motion_detection mode)
            if not self.processing_lookback:
                self.frame_buffer.append((frame_num, frame.copy()))
                # Keep buffer at max size
                if len(self.frame_buffer) > self.lookback_frames:
                    self.frame_buffer.pop(0)
            
            # Fast motion detection mode (ROI only)
            has_motion = self.detect_motion(roi_frame)
            
            if has_motion:
                # Motion detected - switch to tracking mode
                logger.debug(f"Frame {frame_num}: Motion detected, switching to tracking mode")
                self.mode = 'tracking'
                self.frames_since_motion = 0
                
                # Process lookback buffer FIRST (if we have buffered frames)
                if self.frame_buffer and not self.processing_lookback:
                    self.processing_lookback = True  # Prevent infinite loops
                    logger.debug(f"Processing {len(self.frame_buffer)} lookback frames")
                    
                    # The buffer's frames all need YOLO and none of their
                    # detections depend on each other, so this is one batched
                    # forward pass instead of N. The tracker is still fed one
                    # frame at a time, in order — only the detector call is
                    # coalesced, so results are identical.
                    buffered = list(self.frame_buffer)
                    batched = self.yolo_detector.detect_batch(
                        [f for _, f in buffered])

                    for (buf_frame_num, buf_frame), yolo_detections in zip(buffered, batched):
                        buf_detections = to_tracker_rows(yolo_detections)

                        # Update tracker with lookback detections
                        if self.tracker is not None:
                            self._give_frame(buf_frame)
                            buf_tracks = self.tracker.update(buf_detections, buf_frame_num)
                            # These frames are real detections too — the start
                            # of the bee's arrival — so they get crops.
                            if self.save_crops:
                                self._save_track_crops(buf_frame, buf_frame_num)
                            lookback_results.append({
                                'frame_num': buf_frame_num,
                                'detections': buf_detections,
                                'tracks': buf_tracks,
                                'mode': 'lookback'
                            })
                    
                    self.frame_buffer = []  # Clear buffer
                    self.processing_lookback = False
                
                # Now run YOLO on current frame
                yolo_detections = self.yolo_detector.detect(frame)
                
                detections.extend(to_tracker_rows(yolo_detections))
        
        else:  # tracking mode
            # Run YOLO on FULL FRAME (allows tracking beyond ROI)
            yolo_detections = self.yolo_detector.detect(frame)
            
            detections.extend(to_tracker_rows(yolo_detections))
            
            # Check if motion has stopped (ROI only)
            has_motion = self.detect_motion(roi_frame)
            
            if has_motion:
                self.frames_since_motion = 0
            else:
                self.frames_since_motion += 1
            
            # Switch back to motion detection mode if no motion for cooldown period
            if self.frames_since_motion > self.motion_cooldown:
                logger.debug(f"Frame {frame_num}: No motion for {self.motion_cooldown} frames, switching to motion detection mode")
                self.mode = 'motion_detection'
                self.frame_buffer = []  # Clear buffer when switching back
        
        # Update tracker (full frame coordinates, no adjustment needed)
        tracks = []
        if self.tracker is not None:
            self._give_frame(frame)
            tracks = self.tracker.update(detections, frame_num)
            # A track confirmed this frame, at the frames it was seen in while
            # it waited: reported like lookback frames, sorted into place later.
            by_frame = {}
            for bf_frame, row in getattr(self.tracker, 'backfill', None) or []:
                by_frame.setdefault(bf_frame, []).append(row)
            for bf_frame in sorted(by_frame):
                lookback_results.append({
                    'frame_num': bf_frame, 'detections': [], 'tracks': by_frame[bf_frame],
                    'mode': 'tracking', 'num_detections': 0,
                    'num_tracks': len(by_frame[bf_frame]),
                })
            
            # Run identification on confirmed tracks
            if self.identifier and tracks:
                for track_obj in self.tracker.tracks:
                    if track_obj.is_confirmed and track_obj.bee_id is None:
                        # Try to identify this bee
                        result = self.identifier.identify(frame, track_obj.last_bbox)
                        if result:
                            bee_id, method, confidence = result
                            track_obj.set_bee_id(bee_id, method, confidence)
                
                # Refresh tracks list with updated bee_ids
                tracks = self.tracker.get_active_tracks()
        
        # Species classification — every frame, every confirmed track. A track
        # is one animal, so its frames should agree; where they don't the
        # majority wins. That is why this runs on all frames rather than
        # stopping at the first confident read the way markers do.
        if self.species_classifier and tracks:
            self._classify_species(frame, tracks)

        # Save crops for identification training
        if self.save_crops and self.tracker is not None:
            self._save_track_crops(frame, frame_num)
        
        result = {
            'frame_num': frame_num,
            'detections': detections,
            'tracks': tracks,
            'mode': self.mode,
            'num_detections': len(detections),
            'num_tracks': len(tracks),
            'lookback_results': lookback_results  # Include lookback frames
        }
        
        if visualize:
            result['visualization'] = self._create_visualization(frame, detections, tracks)
        
        return result
    
    def _create_visualization(
        self,
        frame: np.ndarray,
        detections: List,
        tracks: List[Dict]
    ) -> np.ndarray:
        """
        Create visualization of detections and tracks.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of tracks
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw ROI — the traced outline when there is one, so the annotated video
        # shows the region that was actually watched.
        if self.roi_polygon:
            cv2.polylines(vis_frame, [np.array(self.roi_polygon, dtype=np.int32)],
                          True, (0, 255, 0), 2)
        elif self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detections (BLUE for YOLO)
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            source = det[5] if len(det) > 5 else 'unknown'
            
            color = (255, 0, 0) if source == 'yolo' else (128, 128, 128)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, source.upper(), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracks
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            track_id = track['track_id']
            cx, cy = int(track['cx']), int(track['cy'])
            
            # Draw trajectory
            if len(track['history']) > 1:
                points = np.array(track['history'], dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)
            
            # Draw current position
            cv2.circle(vis_frame, (cx, cy), 5, color, -1)
            
            # Show bee_id if available, otherwise track_id
            label = f"B:{track['bee_id']}" if track.get('bee_id') else f"T:{track_id}"
            cv2.putText(vis_frame, label, (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(vis_frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process entire video (or one frame range of it).

        Args:
            video_path: Path to input video
            output_path: Path to output video (if visualize=True)
            visualize: Whether to create output video
            start_frame: First frame to process (chunked long videos)
            end_frame: Stop before this frame (exclusive); None = to EOF

        Returns:
            List of results for each frame
        """
        # Initialize for this video
        self.initialize_video(video_path, output_path)

        # Open video
        cap = cv2.VideoCapture(video_path)

        # Frame-range support (chunked long videos): seek to start_frame; the
        # loop stops before end_frame. frame_num stays ABSOLUTE within the
        # original video, so event timestamps (recording_start + frame/fps)
        # and cross-chunk CSV merges line up without any offsetting.
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Chunk range: seeking to frame {start_frame}"
                        + (f", processing until {end_frame}" if end_frame else " (to EOF)"))

        # Setup video writer if visualizing
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.video_width, self.video_height)
            )

        results = []
        frames_read = 0

        frames = _iter_frames(cap, start_frame, end_frame)
        for frame_num, frame in frames:
            frames_read += 1
            # Process frame
            result = self.process_frame(frame, frame_num, visualize=visualize)
            
            # Handle lookback results (frames processed before current frame)
            lookback_results = result.pop('lookback_results', [])
            if lookback_results:
                logger.debug(f"Adding {len(lookback_results)} lookback results")
                results.extend(lookback_results)
            
            # Write visualization if enabled, then DROP the frame image from the
            # result — results are retained for the whole video, and keeping
            # every annotated frame (~2.7 MB each at 720p) OOM-kills the worker
            # after a few thousand frames.
            if visualize and output_path and 'visualization' in result:
                with PROFILER.stage("encode"):
                    out.write(result['visualization'])
            result.pop('visualization', None)

            results.append(result)

            if frames_read % 100 == 0:
                logger.info(f"Processed {frames_read} frames")
                # The owner cancelled this clip (memory/49): stop the reader
                # thread and let go of the file, then unwind. A flag read only;
                # the S3 look-up happens on the worker's watcher thread.
                if cancellation.requested():
                    frames.close()
                    cap.release()
                    if visualize and output_path:
                        out.release()
                    logger.info(f"Cancelled after {frames_read} frames")
                    cancellation.check()

        cap.release()
        if visualize and output_path:
            out.release()
        
        logger.info(f"Processed {frames_read} frames total")
        
        # Log crop saving summary
        if self.save_crops and self.track_crop_counts:
            import os
            total_crops = sum(self.track_crop_counts.values())
            num_tracks = len(self.track_crop_counts)
            abs_crop_dir = os.path.abspath(self.crop_output_dir)
            logger.info(f"Saved {total_crops} crops for {num_tracks} tracks in: {abs_crop_dir}")
            self._write_crop_sharpness()
        
        # Convert results to DataFrame
        import pandas as pd

        # Raw, pre-association detections. Every frame's detector output is
        # already sitting in `results` (process_frame stores it) and was simply
        # discarded — the flatten loop below only reads 'tracks'. Building this
        # costs no extra inference, and it's what "count detections" analyses
        # actually want: the tracked table drops anything the tracker never
        # associated into a confirmed track.
        self.detections_df = self._build_detections_df(results)

        # Define clean column list
        columns = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
                   'confidence', 'source', 'taxon', 'taxon_confidence', 'taxon_votes',
                   'bee_id', 'bee_id_method', 'bee_id_confidence', 'mode']
        
        if not results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=columns)
        
        # Flatten results - one row per track per frame
        flattened_rows = []
        
        for result in results:
            frame_num = result['frame_num']
            mode = result['mode']
            
            # Extract each track into a separate row
            for track in result['tracks']:
                track_id = track.get('track_id')
                
                if track_id is None:
                    logger.warning(f"Track missing track_id! Track keys: {track.keys()}")
                    continue  # Skip tracks without ID

                species = self.species_for(track_id)
                
                row = {
                    'frame': frame_num,
                    'track_id': track_id,
                    'x1': track['x1'],
                    'y1': track['y1'],
                    'x2': track['x2'],
                    'y2': track['y2'],
                    'cx': track['cx'],
                    'cy': track['cy'],
                    'confidence': track.get('confidence', 0.0),
                    'source': track.get('source', 'unknown'),
                    # The voted species wins over the detector's class label:
                    # YOLO says "bee", BeeMachine says which bee. Falls back to
                    # the detector label when no classifier ran or nothing was
                    # legible, so this column is never empty.
                    'taxon': (species[0] if species else track.get('taxon', 'bee')),
                    'taxon_confidence': (species[1] if species else None),
                    'taxon_votes': (species[2] if species else 0),
                    'bee_id': track.get('bee_id'),
                    'bee_id_method': track.get('bee_id_method'),
                    'bee_id_confidence': track.get('bee_id_confidence', 0.0),
                    'mode': mode
                }
                flattened_rows.append(row)
        
        # Create DataFrame from flattened rows
        if not flattened_rows:
            logger.warning("No tracks detected in video")
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(flattened_rows)
        
        # Sort by frame number to ensure correct temporal order (lookback frames may be added later)
        df = df.sort_values(['frame', 'track_id']).reset_index(drop=True)
        
        logger.info(f"Created tracking DataFrame with {len(df)} rows ({len(results)} frames, "
                   f"{len(df['track_id'].unique())} unique tracks)")

        return df

    # Column contract for the detections table (see _build_detections_df).
    DETECTION_COLUMNS = ['frame', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
                         'confidence', 'source', 'taxon', 'mode']

    def _build_detections_df(self, results):
        """Flatten every frame's raw detector output into one DataFrame.

        One row per detection per frame, *before* the tracker associates them —
        so a detection that never became a confirmed track still appears here.
        Detection entries are the flat lists built in ``process_frame``:
        ``[x1, y1, x2, y2, confidence, source, taxon]``. Rows are read
        defensively by position so a detector that emits a shorter list (or a
        future one that appends fields) doesn't break the export.
        """
        import pandas as pd

        rows = []
        for result in results or []:
            frame_num = result.get('frame_num')
            mode = result.get('mode')
            for det in result.get('detections') or []:
                if len(det) < 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in det[:4])
                rows.append({
                    'frame': frame_num,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': (x1 + x2) / 2.0,
                    'cy': (y1 + y2) / 2.0,
                    'confidence': float(det[4]) if len(det) > 4 else 0.0,
                    'source': det[5] if len(det) > 5 else 'unknown',
                    'taxon': det[6] if len(det) > 6 else 'bee',
                    'mode': mode,
                })
        if not rows:
            return pd.DataFrame(columns=self.DETECTION_COLUMNS)
        df = pd.DataFrame(rows).sort_values('frame').reset_index(drop=True)
        logger.info(f"Created detections DataFrame with {len(df)} rows "
                    f"({df['frame'].nunique()} frames with detections)")
        return df