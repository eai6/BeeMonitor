"""Cloud-friendly wrapper around BeeMonitor.analyze_video().

Downloads video from S3 (raw-videos), runs the analysis pipeline, and uploads
results back to S3 (processed). Designed to be called from any cloud compute
environment — SageMaker, Modal (legacy), or local.
"""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from cloud.storage.config import S3Config
from cloud.storage.s3_client import S3StorageClient
from cloud.wrapper.model_manager import ModelManager, ModelPaths

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured result returned after cloud processing."""

    job_id: str
    user_id: str
    total_events: int
    entry_count: int
    exit_count: int
    unique_tracks: int
    nest_count: int
    events_csv_path: str  # S3 key in processed bucket
    tracking_csv_path: str
    foraging_trips_csv_path: str
    interactions_csv_path: str
    annotated_video_path: str  # empty if visualize=False
    foraging_trip_count: int
    avg_trip_duration_sec: float
    interaction_count: int
    summary_stats: dict

    def to_dict(self) -> dict:
        return asdict(self)


class CloudPipeline:
    """Orchestrate BeeMonitor analysis in a cloud environment.

    Workflow:
        1. Download video from S3 (raw-videos bucket)
        2. Ensure ML models are available locally
        3. Run BeeMonitor.analyze_video()
        4. Upload results to S3 (processed bucket)
        5. Return structured PipelineResult
    """

    def __init__(
        self,
        storage_client: Optional[S3StorageClient] = None,
        storage_config: Optional[S3Config] = None,
        model_manager: Optional[ModelManager] = None,
        local_work_dir: str = "/tmp/beemonitor_work",
    ):
        self._config = storage_config or S3Config()
        self._storage = storage_client or S3StorageClient(self._config)
        self._models = model_manager or ModelManager(
            storage_client=self._storage, storage_config=self._config
        )
        self.work_dir = Path(local_work_dir)

    def process(
        self,
        job_id: str,
        user_id: str,
        video_blob_path: str,
        detection_mode: str = "yolo",
        confidence_threshold: float = 0.25,
        ml_threshold: float = 0.6,
        visualize: bool = True,
        two_mode_tracking: bool = True,
        custom_nest_model_path: str = "",
        custom_bee_model_path: str = "",
        hotel_roi=None,
        nest_layout=None,
        run_tracking: bool = True,
    ) -> PipelineResult:
        """Run the full BeeMonitor pipeline on a video stored in S3.

        Args:
            job_id: Unique job identifier.
            user_id: Owner of the video.
            video_blob_path: Key within the raw-videos bucket.
            detection_mode: 'yolo' (default) or 'yolo_only'.
            confidence_threshold: YOLO detection threshold.
            ml_threshold: Event classifier threshold.
            visualize: Whether to generate annotated video.
            two_mode_tracking: Use blob motion detection + YOLO (faster) vs YOLO every frame.

        Returns:
            PipelineResult with S3 keys to all outputs.
        """
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Step 1 — Download video from S3 raw-videos
        video_local = str(job_dir / Path(video_blob_path).name)
        logger.info("[%s] Downloading video: %s", job_id, video_blob_path)
        self._storage.download_file("raw-videos", video_blob_path, video_local)

        # Step 2 — Ensure models
        logger.info("[%s] Ensuring models are available", job_id)
        model_paths = self._models.ensure_models()

        # Step 2b — Download custom models if specified
        custom_nest_local = ""
        custom_bee_local = ""
        if custom_nest_model_path:
            logger.info("[%s] Downloading custom nest model: %s", job_id, custom_nest_model_path)
            custom_nest_local = self._models.ensure_custom_model(custom_nest_model_path)
        if custom_bee_model_path:
            logger.info("[%s] Downloading custom bee model: %s", job_id, custom_bee_model_path)
            custom_bee_local = self._models.ensure_custom_model(custom_bee_model_path)

        # Nest/hotel-only jobs (run_tracking=False, e.g. the pipeline builder's
        # Detect Nest block) skip motion detection, tracking, events, and all
        # post-processing — they only need the layout.
        if not run_tracking:
            return self._nest_only(job_id, user_id, video_local, model_paths,
                                   custom_nest_local, hotel_roi, nest_layout)

        # Step 3 — Build config and run analysis
        logger.info("[%s] Running BeeMonitor analysis", job_id)
        result = self._run_analysis(
            video_local=video_local,
            output_dir=str(output_dir),
            model_paths=model_paths,
            detection_mode=detection_mode,
            confidence_threshold=confidence_threshold,
            ml_threshold=ml_threshold,
            visualize=visualize,
            two_mode_tracking=two_mode_tracking,
            custom_nest_local=custom_nest_local,
            custom_bee_local=custom_bee_local,
            hotel_roi=hotel_roi,
            nest_layout=nest_layout,
        )

        # Step 4 — Post-processing: foraging trips + interactions
        events = result.events if result is not None else None
        tracks = result.tracks if result is not None else None
        nests = result.nests if result is not None else None

        from cloud.wrapper.foraging import compute_foraging_trips, compute_trip_summary

        # Detect fps from config or default to 30
        fps = 30.0
        if result and hasattr(result, "config") and result.config:
            try:
                fps = float(result.config.video.fps)
            except Exception:
                pass

        trips_df = compute_foraging_trips(events, fps=fps)
        trip_summary = compute_trip_summary(trips_df)

        # Save foraging trips CSV to output dir
        if not trips_df.empty:
            trips_csv_path = str(output_dir / "foraging_trips.csv")
            trips_df.to_csv(trips_csv_path, index=False)
            logger.info("[%s] Saved %d foraging trips", job_id, len(trips_df))

        # Step 4b — Compute interactions from tracking data
        interaction_count = 0
        try:
            if (tracks is not None and hasattr(tracks, "empty") and not tracks.empty
                    and nests and isinstance(nests, dict) and nests.get("nests")):
                from beemonitor.processing.interaction_analyzer import InteractionAnalyzer, nests_to_reference_objects

                analyzer = InteractionAnalyzer(fps=fps)

                # Bee-to-bee interactions
                track_interactions, _ = analyzer.analyze_track_interactions(tracks)

                # Bee-to-reference (nest) interactions
                ref_objects = nests_to_reference_objects(
                    [{"id": k, "bbox": v} for k, v in nests["nests"].items()]
                )
                ref_interactions, _ = analyzer.analyze_reference_interactions(tracks, ref_objects)

                all_interactions = track_interactions + ref_interactions
                interaction_count = len(all_interactions)

                if all_interactions:
                    # Build track_id → taxon lookup from tracking data
                    import pandas as pd
                    taxon_lookup = {}
                    if tracks is not None and "taxon" in tracks.columns and "track_id" in tracks.columns:
                        for tid, grp in tracks.groupby("track_id"):
                            taxon_lookup[tid] = grp["taxon"].mode().iloc[0] if len(grp) > 0 else "unknown"

                    # Combined CSV — organism-to-organism and organism-to-reference
                    # Each row is one pairwise interaction. Multi-party encounters
                    # produce multiple rows (e.g., A near B and C = rows A-B, A-C, B-C)
                    rows = []
                    track_set = set(id(e) for e in track_interactions)
                    for event in all_interactions:
                        is_track = id(event) in track_set
                        rows.append({
                            "interaction_type": "organism-to-organism" if is_track else "organism-to-reference",
                            "organism_track_id": event.entity1_id,
                            "organism_taxon": taxon_lookup.get(event.entity1_id, "unknown"),
                            "partner_track_id": event.entity2_id if is_track else "",
                            "partner_taxon": taxon_lookup.get(event.entity2_id, "") if is_track else "",
                            "reference_id": "" if is_track else event.entity2_id,
                            "start_frame": event.start_frame,
                            "end_frame": event.end_frame,
                            "duration_frames": event.duration_frames,
                            "duration_seconds": round(event.duration_frames / fps, 2),
                            "min_distance_px": round(event.min_distance, 1),
                            "avg_distance_px": round(event.avg_distance, 1),
                        })
                    pd.DataFrame(rows).to_csv(str(output_dir / "interactions.csv"), index=False)
                    logger.info("[%s] Saved %d interactions (%d organism-to-organism, %d organism-to-reference)",
                                job_id, len(all_interactions), len(track_interactions), len(ref_interactions))
        except Exception as e:
            logger.warning("[%s] Interaction analysis failed (non-fatal): %s", job_id, e)

        # Step 5 — Upload results to S3 processed bucket
        logger.info("[%s] Uploading results to S3", job_id)
        result_paths = self._upload_results(job_id, user_id, output_dir, video_local)

        # Step 6 — Build structured result

        # get_statistics() exists on analysis_results.AnalysisResults but not
        # on the version defined in video_analyzer.py — handle both
        if result and hasattr(result, "get_statistics"):
            stats = result.get_statistics()
        elif result and events is not None and hasattr(events, "empty") and not events.empty:
            import pandas as pd
            stats = {
                "total_events": len(events),
                "total_entries": int((events["action"] == "Entry").sum()) if "action" in events.columns else 0,
                "total_exits": int((events["action"] == "Exit").sum()) if "action" in events.columns else 0,
                "total_tracks": int(tracks["track_id"].nunique()) if tracks is not None and "track_id" in tracks.columns else 0,
                "total_nests": int(events["nest"].nunique()) if "nest" in events.columns else 0,
            }
        else:
            stats = {}

        # Merge foraging trip + interaction summary into stats
        stats["foraging_trips"] = trip_summary
        stats["video_fps"] = fps
        stats["interaction_count"] = interaction_count

        # Persist nest bounding boxes for future use (avoids re-detection)
        if nests and isinstance(nests, dict) and nests.get("nests"):
            # Convert numpy arrays/tuples to plain lists for JSON serialization
            nest_bboxes = {}
            for nest_id, bbox in nests["nests"].items():
                nest_bboxes[str(nest_id)] = [float(x) for x in bbox]
            stats["nest_bboxes"] = nest_bboxes

        total_events = len(events) if events is not None and hasattr(events, "__len__") else 0
        unique_tracks = 0
        if tracks is not None and hasattr(tracks, "nunique") and "track_id" in tracks.columns:
            unique_tracks = int(tracks["track_id"].nunique())

        pipeline_result = PipelineResult(
            job_id=job_id,
            user_id=user_id,
            total_events=total_events,
            entry_count=int(stats.get("total_entries", 0)),
            exit_count=int(stats.get("total_exits", 0)),
            unique_tracks=unique_tracks,
            nest_count=int(stats.get("total_nests", 0)),
            events_csv_path=result_paths.get("events_csv", ""),
            tracking_csv_path=result_paths.get("tracking_csv", ""),
            foraging_trips_csv_path=result_paths.get("foraging_trips_csv", ""),
            interactions_csv_path=result_paths.get("interactions_csv", ""),
            annotated_video_path=result_paths.get("annotated_video", ""),
            foraging_trip_count=trip_summary["total_trips"],
            avg_trip_duration_sec=trip_summary["avg_duration_sec"],
            interaction_count=interaction_count,
            summary_stats=stats,
        )

        logger.info("[%s] Pipeline complete — %d events, %d foraging trips",
                     job_id, pipeline_result.total_events, pipeline_result.foraging_trip_count)
        return pipeline_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nest_only(self, job_id, user_id, video_local, model_paths,
                   custom_nest_local="", hotel_roi=None, nest_layout=None):
        """Fast path for run_tracking=False: the device layout if present,
        else NestDetector on the video's early frames. No motion detection,
        tracking model, events, or post-processing."""
        nests = self._build_manual_nests(video_local, hotel_roi, nest_layout)
        if nests is None:
            from ultralytics import YOLO
            from beemonitor.core.config import Config
            from beemonitor.detection.nest_detector import NestDetector

            config = Config.default()
            config.models.nest_detection = custom_nest_local or model_paths.nest_detection
            detector = NestDetector(model=YOLO(config.models.nest_detection), config=config)
            nests = detector.get_nests_and_hotel_detections(video_path=video_local)

        nest_bboxes = {}
        for nest_id, bbox in ((nests or {}).get("nests") or {}).items():
            nest_bboxes[str(nest_id)] = [float(x) for x in bbox]
        hotel = (nests or {}).get("hotel")

        preview_key = ""
        try:
            preview_key = self._upload_nest_preview(job_id, user_id, video_local, nests)
        except Exception as e:  # preview is a nicety — never fail the job over it
            logger.warning("[%s] nest preview render failed (non-fatal): %s", job_id, e)

        stats = {
            "total_nests": len(nest_bboxes),
            "nest_bboxes": nest_bboxes,
            "nest_preview_path": preview_key,
            "nest_only": True,
        }
        if hotel:
            stats["hotel_bbox"] = [float(x) for x in hotel]

        logger.info("[%s] Nest-only job complete — %d nests", job_id, len(nest_bboxes))
        return PipelineResult(
            job_id=job_id, user_id=user_id,
            total_events=0, entry_count=0, exit_count=0,
            unique_tracks=0, nest_count=len(nest_bboxes),
            events_csv_path="", tracking_csv_path="", foraging_trips_csv_path="",
            interactions_csv_path="", annotated_video_path="",
            foraging_trip_count=0, avg_trip_duration_sec=0.0, interaction_count=0,
            summary_stats=stats,
        )

    def _upload_nest_preview(self, job_id, user_id, video_local, nests):
        """Draw the hotel ROI (orange) + nest boxes (green, labeled) on the
        video's first frame and upload a JPEG to the processed bucket.
        Returns the S3 key, or '' when no frame could be read."""
        import cv2

        cap = cv2.VideoCapture(video_local)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return ""

        hotel = (nests or {}).get("hotel")
        if hotel:
            x1, y1, x2, y2 = [int(v) for v in hotel]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 255), 3)
            cv2.putText(frame, "hotel", (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2)
        for nest_id, bbox in ((nests or {}).get("nests") or {}).items():
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, str(nest_id), (x1, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

        local = str(Path(video_local).with_name("nest_preview.jpg"))
        cv2.imwrite(local, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        key = f"{user_id}/{job_id}/nest_preview.jpg"
        self._storage.upload_file("processed", key, local)
        return key

    def _run_analysis(
        self,
        video_local: str,
        output_dir: str,
        model_paths: ModelPaths,
        detection_mode: str,
        confidence_threshold: float,
        ml_threshold: float,
        visualize: bool,
        two_mode_tracking: bool = True,
        custom_nest_local: str = "",
        custom_bee_local: str = "",
        hotel_roi=None,
        nest_layout=None,
    ):
        """Build a BeeMonitor Config, instantiate, and run."""
        from beemonitor.core.config import Config, ModelConfig

        config = Config.default()

        # Point to downloaded models
        config.models.nest_detection = model_paths.nest_detection
        config.models.tracking = model_paths.bee_tracking
        config.models.event_classifier = model_paths.event_classifier

        # Override with custom models if provided (separate for nest and bee)
        if custom_nest_local:
            logger.info("Using custom nest model: %s", custom_nest_local)
            config.models.nest_detection = custom_nest_local
        if custom_bee_local:
            logger.info("Using custom bee model: %s", custom_bee_local)
            config.models.tracking = custom_bee_local

        # Apply user overrides
        config.tracking.confidence_threshold = confidence_threshold
        config.tracking.ml_threshold = ml_threshold
        config.tracking.enable_two_mode_tracking = two_mode_tracking
        config.output.save_visualizations = visualize

        # Device-supplied hotel ROI + nest tubes (normalized 0..1) → pixel-space
        # manual_nests so the run uses the human-set layout (model is the backup).
        # Requires BOTH the ROI and the tubes; otherwise fall back to detection.
        manual_nests = self._build_manual_nests(video_local, hotel_roi, nest_layout)

        from beemonitor import BeeMonitor

        monitor = BeeMonitor(config=config)
        result = monitor.analyze_video(
            video_path=video_local,
            output_folder=output_dir,
            visualize=visualize,
            detection_mode=detection_mode,
            manual_nests=manual_nests,
        )
        return result

    @staticmethod
    def _build_manual_nests(video_local: str, hotel_roi, nest_layout):
        """Turn a device's normalized hotel ROI + nest layout into the pixel-space
        ``{'hotel': (x1,y1,x2,y2), 'nests': {id: (x1,y1,x2,y2)}}`` the analyzer
        consumes. Returns None (→ model detection) unless BOTH are present + valid."""
        if not hotel_roi or not nest_layout:
            return None
        try:
            import cv2
            cap = cv2.VideoCapture(video_local)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if not (w and h):
                return None

            def to_px(box):
                return (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))

            nests = {}
            for item in nest_layout:
                box = item.get("box")
                if box and len(box) == 4:
                    nests[item.get("id")] = to_px(box)
            if not nests:
                return None
            logger.info("Using device hotel ROI + %d nest tubes (manual layout)", len(nests))
            return {"hotel": to_px(hotel_roi), "nests": nests}
        except Exception as e:  # never let layout parsing break the run — fall back
            logger.warning("manual_nests build failed (%s) — falling back to detection", e)
            return None

    def _upload_results(
        self, job_id: str, user_id: str, output_dir: Path, video_local: str
    ) -> dict[str, str]:
        """Upload all result files from output_dir to the S3 processed bucket."""
        container = "processed"
        prefix = f"{user_id}/{job_id}"
        uploaded: dict[str, str] = {}

        video_stem = Path(video_local).stem

        # Events CSV
        events_files = list(output_dir.glob(f"*_events.csv"))
        if events_files:
            blob_path = f"{prefix}/events.csv"
            self._storage.upload_file(container, blob_path, str(events_files[0]))
            uploaded["events_csv"] = blob_path

        # Tracking CSV
        tracking_files = list(output_dir.glob(f"*_tracking_results.csv"))
        if tracking_files:
            blob_path = f"{prefix}/tracking_results.csv"
            self._storage.upload_file(container, blob_path, str(tracking_files[0]))
            uploaded["tracking_csv"] = blob_path

        # Foraging trips CSV
        trips_files = list(output_dir.glob("foraging_trips.csv"))
        if trips_files:
            blob_path = f"{prefix}/foraging_trips.csv"
            self._storage.upload_file(container, blob_path, str(trips_files[0]))
            uploaded["foraging_trips_csv"] = blob_path

        # Interactions CSV
        interactions_files = list(output_dir.glob("interactions.csv"))
        if interactions_files:
            blob_path = f"{prefix}/interactions.csv"
            self._storage.upload_file(container, blob_path, str(interactions_files[0]))
            uploaded["interactions_csv"] = blob_path

        # Annotated video — re-encode to H.264 with ffmpeg for browser playback
        video_files = list(output_dir.glob("*.mp4"))
        if video_files:
            src = str(video_files[0])
            h264 = str(output_dir / "annotated_h264.mp4")
            try:
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src, "-c:v", "libx264",
                     "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", "-an", h264],
                    check=True, capture_output=True, timeout=600,
                )
                upload_file = h264
                logger.info("[%s] Re-encoded to H.264", job_id)
            except Exception as e:
                logger.warning("[%s] ffmpeg failed, uploading mp4v: %s", job_id, e)
                upload_file = src

            blob_path = f"{prefix}/annotated_video.mp4"
            self._storage.upload_file(
                container, blob_path, upload_file, content_type="video/mp4"
            )
            uploaded["annotated_video"] = blob_path

        # Job metadata
        metadata = {
            "job_id": job_id,
            "user_id": user_id,
            "source_video": video_local,
            "files_uploaded": uploaded,
        }
        meta_path = output_dir / "job_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        self._storage.upload_file(
            container, f"{prefix}/job_metadata.json", str(meta_path)
        )

        return uploaded

    def cleanup(self, job_id: str) -> None:
        """Remove local working files for a completed job."""
        import shutil

        job_dir = self.work_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info("[%s] Cleaned up local files", job_id)
