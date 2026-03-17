"""CPU function for post-processing: CSV generation and video annotation.

Can be used separately when video annotation is requested after initial
processing, or to regenerate results with different parameters.
"""

import logging
import tempfile
from pathlib import Path

import modal

from cloud.modal_app.app import app
from cloud.modal_app.secrets import azure_secret

logger = logging.getLogger(__name__)


@app.function(
    timeout=3600,
    secrets=[azure_secret],
    memory=8192,
    cpu=4.0,
)
def generate_annotated_video(
    job_id: str,
    user_id: str,
    video_blob_path: str,
    events_blob_path: str,
    tracking_blob_path: str,
) -> dict:
    """Generate an annotated video from existing processing results.

    Used when the original job ran with visualize=False and the user
    later requests a visualization.

    Args:
        job_id: Job identifier.
        user_id: User identifier.
        video_blob_path: Raw video path in Azure.
        events_blob_path: Events CSV path in processed container.
        tracking_blob_path: Tracking CSV path in processed container.

    Returns:
        Dict with annotated_video_path in Azure.
    """
    import pandas as pd

    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig

    config = StorageConfig()
    storage = AzureBlobClient(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download source files
        video_local = str(tmpdir / "video.mp4")
        storage.download_file(config.raw_videos_container, video_blob_path, video_local)

        events_local = str(tmpdir / "events.csv")
        storage.download_file(config.processed_container, events_blob_path, events_local)

        tracking_local = str(tmpdir / "tracking.csv")
        storage.download_file(config.processed_container, tracking_blob_path, tracking_local)

        # Load data
        events_df = pd.read_csv(events_local)
        tracking_df = pd.read_csv(tracking_local)

        # Generate annotated video
        from beemonitor.core.config import Config
        from beemonitor.output.video_synthesizer import VideoSynthesizer

        bm_config = Config.default()
        synthesizer = VideoSynthesizer(bm_config)

        output_dir = str(tmpdir / "output")
        Path(output_dir).mkdir()
        output_path = synthesizer.synthesize(
            video_local, events_df, tracking_df, {}, output_dir
        )

        # Upload annotated video
        blob_path = f"{user_id}/{job_id}/annotated_video.mp4"
        storage.upload_file(
            config.processed_container,
            blob_path,
            output_path,
            content_type="video/mp4",
        )

    return {"annotated_video_path": blob_path}
