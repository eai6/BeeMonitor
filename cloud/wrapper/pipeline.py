"""Cloud-friendly wrapper around BeeMonitor.analyze_video().

Downloads video from Azure, runs the analysis pipeline, and uploads results
back to Azure Blob Storage. Designed to be called from Modal serverless
functions or any cloud compute environment.
"""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from cloud.storage.azure_client import AzureBlobClient
from cloud.storage.config import StorageConfig
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
    events_csv_path: str  # Azure blob path
    tracking_csv_path: str  # Azure blob path
    annotated_video_path: str  # Azure blob path (empty if visualize=False)
    summary_stats: dict

    def to_dict(self) -> dict:
        return asdict(self)


class CloudPipeline:
    """Orchestrate BeeMonitor analysis in a cloud environment.

    Workflow:
        1. Download video from Azure (raw-videos container)
        2. Ensure ML models are available locally
        3. Run BeeMonitor.analyze_video()
        4. Upload results to Azure (processed container)
        5. Return structured PipelineResult
    """

    def __init__(
        self,
        storage_client: Optional[AzureBlobClient] = None,
        storage_config: Optional[StorageConfig] = None,
        model_manager: Optional[ModelManager] = None,
        local_work_dir: str = "/tmp/beemonitor_work",
    ):
        self._config = storage_config or StorageConfig()
        self._storage = storage_client or AzureBlobClient(self._config)
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
    ) -> PipelineResult:
        """Run the full BeeMonitor pipeline on a video stored in Azure.

        Args:
            job_id: Unique job identifier.
            user_id: Owner of the video.
            video_blob_path: Path within the raw-videos container.
            detection_mode: 'yolo' (default) or 'yolo_only'.
            confidence_threshold: YOLO detection threshold.
            ml_threshold: Event classifier threshold.
            visualize: Whether to generate annotated video.

        Returns:
            PipelineResult with Azure paths to all outputs.
        """
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Step 1 — Download video
        video_local = str(job_dir / Path(video_blob_path).name)
        logger.info("[%s] Downloading video: %s", job_id, video_blob_path)
        self._storage.download_file(
            self._config.raw_videos_container, video_blob_path, video_local
        )

        # Step 2 — Ensure models
        logger.info("[%s] Ensuring models are available", job_id)
        model_paths = self._models.ensure_models()

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
        )

        # Step 4 — Upload results to Azure
        logger.info("[%s] Uploading results to Azure", job_id)
        azure_paths = self._upload_results(job_id, user_id, output_dir, video_local)

        # Step 5 — Build structured result
        events = result.events if result is not None else None
        tracks = result.tracks if result is not None else None

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
            events_csv_path=azure_paths.get("events_csv", ""),
            tracking_csv_path=azure_paths.get("tracking_csv", ""),
            annotated_video_path=azure_paths.get("annotated_video", ""),
            summary_stats=stats,
        )

        logger.info("[%s] Pipeline complete — %d events", job_id, pipeline_result.total_events)
        return pipeline_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        video_local: str,
        output_dir: str,
        model_paths: ModelPaths,
        detection_mode: str,
        confidence_threshold: float,
        ml_threshold: float,
        visualize: bool,
    ):
        """Build a BeeMonitor Config, instantiate, and run."""
        from beemonitor.core.config import Config, ModelConfig

        config = Config.default()

        # Point to downloaded models
        config.models.nest_detection = model_paths.nest_detection
        config.models.tracking = model_paths.bee_tracking
        config.models.event_classifier = model_paths.event_classifier

        # Apply user overrides
        config.tracking.confidence_threshold = confidence_threshold
        config.output.save_visualizations = visualize

        from beemonitor import BeeMonitor

        monitor = BeeMonitor(config=config)
        result = monitor.analyze_video(
            video_path=video_local,
            output_folder=output_dir,
            visualize=visualize,
            detection_mode=detection_mode,
        )
        return result

    def _upload_results(
        self, job_id: str, user_id: str, output_dir: Path, video_local: str
    ) -> dict[str, str]:
        """Upload all result files from output_dir to Azure processed container."""
        container = self._config.processed_container
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
