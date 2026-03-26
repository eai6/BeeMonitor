"""
Cloud Analysis Threads
=======================
QThread subclasses for cloud-based GPU video analysis.
"""

import os
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .cloud_client import BeeMonitorCloudClient

# Result file fields to download
_RESULT_FILE_FIELDS = [
    "events_csv_path",
    "tracking_csv_path",
    "foraging_trips_csv_path",
    "interactions_csv_path",
    "annotated_video_path",
]

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class CloudAnalysisThread(QThread):
    """Background thread for single-video cloud analysis."""

    progress = pyqtSignal(str)
    upload_progress = pyqtSignal(int)
    download_progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # output folder path
    error = pyqtSignal(str)

    def __init__(
        self,
        client: BeeMonitorCloudClient,
        video_path: str,
        output_folder: str,
        config: dict | None = None,
    ):
        super().__init__()
        self.client = client
        self.video_path = video_path
        self.output_folder = output_folder
        self.config = config or {}
        self._stop_flag = False

    def run(self):
        try:
            video_name = Path(self.video_path).name

            # Step 1: Upload
            self.progress.emit(f"Uploading {video_name}...")
            video_data = self.client.upload_video(
                self.video_path,
                on_progress=lambda pct: self.upload_progress.emit(pct),
            )
            video_id = video_data["id"]
            self.progress.emit(f"Upload complete (video_id={video_id})")

            if self._stop_flag:
                return

            # Step 2: Create + submit job
            self.progress.emit("Submitting analysis job to cloud GPU...")
            job = self.client.create_and_submit_job(video_id, self.config)
            job_id = job["id"]
            self.progress.emit(f"Job {job_id} submitted (status: {job['status']})")

            # Step 3: Poll until terminal
            while job["status"] not in _TERMINAL_STATUSES:
                if self._stop_flag:
                    return
                time.sleep(15)
                job = self.client.poll_job(job_id)
                pct = job.get("progress_pct", 0)
                self.progress.emit(f"Processing... {pct}% (status: {job['status']})")

            if job["status"] != "completed":
                self.error.emit(
                    f"Job {job_id} {job['status']}: "
                    f"{job.get('error_message', 'Unknown error')}"
                )
                return

            # Step 4: Download results
            self.progress.emit("Downloading results...")
            result = job.get("result") or {}
            os.makedirs(self.output_folder, exist_ok=True)

            downloaded = 0
            for field in _RESULT_FILE_FIELDS:
                if self._stop_flag:
                    return
                blob_path = result.get(field, "")
                if not blob_path:
                    continue

                url = self.client.get_download_url(job_id, field)
                if not url:
                    continue

                local_name = Path(blob_path).name
                local_path = os.path.join(self.output_folder, local_name)
                self.progress.emit(f"Downloading {local_name}...")
                self.client.download_file(
                    url,
                    local_path,
                    on_progress=lambda pct: self.download_progress.emit(pct),
                )
                downloaded += 1

            self.progress.emit(f"Cloud analysis complete! ({downloaded} files downloaded)")
            self.finished.emit(self.output_folder)

        except Exception as e:
            self.error.emit(f"Cloud analysis failed: {e}\n\n{traceback.format_exc()}")

    def stop(self):
        self._stop_flag = True
        self.progress.emit("Stopping cloud analysis...")


class CloudBatchAnalysisThread(QThread):
    """Background thread for batch cloud analysis of a folder.

    Mirrors the web app's batch flow:
    1. Upload all videos to the cloud
    2. Single batch_submit call — server chunks videos across GPU containers,
       models loaded once per chunk (same as BatchJobView)
    3. Poll all jobs until terminal
    4. Download results for completed jobs
    """

    progress = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # (completed, total)
    upload_progress = pyqtSignal(int)
    finished = pyqtSignal(dict)  # {filename: output_folder or None}
    error = pyqtSignal(str)

    def __init__(
        self,
        client: BeeMonitorCloudClient,
        folder_path: str,
        output_folder: str,
        config: dict | None = None,
    ):
        super().__init__()
        self.client = client
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.config = config or {}
        self._stop_flag = False

    def run(self):
        try:
            video_files = sorted(
                f
                for f in os.listdir(self.folder_path)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            )
            total = len(video_files)
            if total == 0:
                self.error.emit("No video files found in folder")
                return

            self.progress.emit(f"Found {total} videos to process in cloud")
            self.progress_update.emit(0, total)

            # Phase A: Upload all videos
            video_ids: dict[str, int] = {}  # filename -> video_id
            for i, vf in enumerate(video_files):
                if self._stop_flag:
                    break
                path = os.path.join(self.folder_path, vf)
                self.progress.emit(f"Uploading {vf} ({i + 1}/{total})...")
                data = self.client.upload_video(
                    path,
                    on_progress=lambda pct: self.upload_progress.emit(pct),
                )
                video_ids[vf] = data["id"]

            if self._stop_flag:
                self.finished.emit({})
                return

            # Phase B: Batch submit (single API call — server chunks across GPUs)
            self.progress.emit(
                f"Submitting batch of {len(video_ids)} videos to cloud GPUs..."
            )
            all_ids = list(video_ids.values())
            batch_resp = self.client.batch_submit(all_ids, self.config)

            submitted = batch_resp.get("submitted", 0)
            skipped = batch_resp.get("skipped", 0)
            job_list = batch_resp.get("jobs", [])

            if skipped:
                self.progress.emit(f"Skipped {skipped} already-analyzed video(s)")
            self.progress.emit(
                f"Submitted {submitted} job(s) — chunked across GPU containers"
            )

            if not job_list:
                self.progress.emit("No new jobs to process")
                self.finished.emit({})
                return

            # Build lookup: video_id -> filename, job_id -> filename
            vid_to_file = {vid: vf for vf, vid in video_ids.items()}
            jobs: dict[str, dict] = {}  # filename -> latest job dict
            for j in job_list:
                vf = vid_to_file.get(j["video_id"], f"video_{j['video_id']}")
                jobs[vf] = {"id": j["id"], "status": j["status"]}
                self.progress.emit(f"  Job {j['id']} -> {vf}")

            # Phase C: Poll until all terminal
            poll_total = len(jobs)
            completed_count = 0
            while not self._stop_flag:
                all_done = True
                for vf, job in list(jobs.items()):
                    if job["status"] in _TERMINAL_STATUSES:
                        continue
                    all_done = False
                    updated = self.client.poll_job(job["id"])
                    if (
                        updated["status"] in _TERMINAL_STATUSES
                        and job["status"] not in _TERMINAL_STATUSES
                    ):
                        completed_count += 1
                        self.progress_update.emit(completed_count, poll_total)
                        status_label = (
                            "Completed" if updated["status"] == "completed" else "Failed"
                        )
                        self.progress.emit(
                            f"{status_label}: {vf} ({completed_count}/{poll_total})"
                        )
                    jobs[vf] = updated
                if all_done:
                    break
                time.sleep(15)

            # Phase D: Download results for completed jobs
            results: dict[str, str | None] = {}
            for vf, job in jobs.items():
                if self._stop_flag:
                    break
                if job["status"] == "completed":
                    result = job.get("result") or {}
                    vf_output = os.path.join(self.output_folder, Path(vf).stem)
                    os.makedirs(vf_output, exist_ok=True)

                    for field in _RESULT_FILE_FIELDS:
                        blob_path = result.get(field, "")
                        if not blob_path:
                            continue
                        url = self.client.get_download_url(job["id"], field)
                        if not url:
                            continue
                        local = os.path.join(vf_output, Path(blob_path).name)
                        self.progress.emit(
                            f"Downloading {Path(blob_path).name} for {vf}..."
                        )
                        self.client.download_file(url, local)

                    results[vf] = vf_output
                else:
                    results[vf] = None

            successful = sum(1 for v in results.values() if v is not None)
            self.progress.emit(
                f"Cloud batch complete: {successful}/{poll_total} videos processed"
            )
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(
                f"Cloud batch analysis failed: {e}\n\n{traceback.format_exc()}"
            )

    def stop(self):
        self._stop_flag = True
        self.progress.emit("Stopping cloud batch analysis...")
