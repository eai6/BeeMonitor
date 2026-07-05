"""BeeMonitor SageMaker **training** entrypoint (BYOC).

Runs inside the training image as `docker run <image> train` (via the `train`
wrapper). SageMaker provides:

  * the input "manifest" channel at /opt/ml/input/data/manifest/manifest.json
  * a writable model dir /opt/ml/model  (tarred to model.tar.gz on success)
  * a failure file path /opt/ml/output/failure  (its contents become the
    DescribeTrainingJob FailureReason the Django poller surfaces)

The manifest (written by Django from the annotation project) tells us which
frames of which videos to train on, plus the hyperparameters. We:

  1. download each video from the raw-videos bucket,
  2. extract the annotated frames + write YOLO label .txt files (80/20 split),
  3. train `YOLO(base_model)` on the assembled dataset,
  4. upload best.pt to the models bucket at the requested key, and
  5. write a result.json (metrics + storage_key) to the output bucket for the
     Django poller — mirroring how the inference container hands back results.

Bucket names come from the container Environment (set on the training job),
matching the inference container's convention.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import boto3
import cv2

MANIFEST = Path("/opt/ml/input/data/manifest/manifest.json")
MODEL_DIR = Path("/opt/ml/model")
FAILURE_FILE = Path("/opt/ml/output/failure")
WORK = Path("/tmp/bm_train")

RAW_VIDEOS_BUCKET = os.environ.get("AWS_S3_BUCKET_RAW_VIDEOS", "")
MODELS_BUCKET = os.environ.get("AWS_S3_BUCKET_MODELS", "")
OUTPUT_BUCKET = os.environ.get("SAGEMAKER_OUTPUT_BUCKET", "")
REGION = os.environ.get("AWS_REGION", "us-east-1")

_s3 = boto3.client("s3", region_name=REGION)


def log(msg: str) -> None:
    print(f"train: {msg}", flush=True)


def _build_dataset(manifest: dict, ds_dir: Path) -> Path:
    """Download videos, extract annotated frames + labels into a YOLO tree.

    Returns the path to the written data.yaml.
    """
    classes = manifest["classes"]
    samples = []  # (filename, label_text, video_local, frame_number)

    vids_dir = WORK / "videos"
    vids_dir.mkdir(parents=True, exist_ok=True)

    for va in manifest["video_annotations"]:
        blob = va["video_blob_path"]
        local = vids_dir / (blob.replace("/", "_"))
        if not local.exists():
            log(f"downloading s3://{RAW_VIDEOS_BUCKET}/{blob}")
            _s3.download_file(RAW_VIDEOS_BUCKET, blob, str(local))
        for fr in va["frames"]:
            samples.append((fr["filename"], fr["label"], local, int(fr["frame_number"])))

    if not samples:
        raise RuntimeError("manifest has no annotated frames to train on")

    # Deterministic 80/20 split (stable across reruns): last 20% -> val.
    samples.sort(key=lambda s: s[0])
    n_val = max(1, len(samples) // 5)
    val_set = set(range(len(samples) - n_val, len(samples)))
    # With very few samples, reuse train as val so YOLO has a val set.
    tiny = len(samples) < 5

    for split in ("train", "val"):
        (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    extracted = 0
    caps: dict = {}
    for i, (filename, label, video_local, frame_number) in enumerate(samples):
        cap = caps.get(str(video_local))
        if cap is None:
            cap = cv2.VideoCapture(str(video_local))
            caps[str(video_local)] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok or frame is None:
            log(f"  skip {filename}: could not read frame {frame_number}")
            continue
        stem = Path(filename).stem
        targets = ["train", "val"] if tiny else (["val"] if i in val_set else ["train"])
        for split in targets:
            cv2.imwrite(str(ds_dir / "images" / split / f"{stem}.jpg"), frame)
            (ds_dir / "labels" / split / f"{stem}.txt").write_text(label or "")
        extracted += 1
    for cap in caps.values():
        cap.release()

    log(f"assembled {extracted}/{len(samples)} frames "
        f"({'tiny: train==val' if tiny else 'train/val 80/20'})")

    data_yaml = ds_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    data_yaml.write_text(
        f"path: {ds_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names:\n{names_block}\n"
    )
    return data_yaml


def _train(manifest: dict, data_yaml: Path) -> dict:
    from ultralytics import YOLO

    # Fine-tune from existing weights (models bucket) when the manifest asks
    # for it; otherwise start from the pretrained base architecture.
    init_key = manifest.get("init_weights_key") or ""
    if init_key:
        base_path = WORK / "init_weights.pt"
        log(f"fine-tuning from s3://{MODELS_BUCKET}/{init_key}")
        _s3.download_file(MODELS_BUCKET, init_key, str(base_path))
        base = str(base_path)
    else:
        base = manifest.get("base_model", "yolov8n")
        if not base.endswith(".pt"):
            base = f"{base}.pt"
    epochs = int(manifest.get("epochs", 50))
    imgsz = int(manifest.get("imgsz", 640))
    batch = int(manifest.get("batch_size", 16))

    log(f"training {base} epochs={epochs} imgsz={imgsz} batch={batch}")
    model = YOLO(base)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        project=str(WORK / "runs"), name="train", exist_ok=True, verbose=True,
    )

    # Locate best.pt.
    best = Path(WORK / "runs" / "train" / "weights" / "best.pt")
    if not best.exists():
        last = Path(WORK / "runs" / "train" / "weights" / "last.pt")
        best = last if last.exists() else None
    if best is None:
        raise RuntimeError("training produced no weights file")

    rd = getattr(results, "results_dict", {}) or {}

    def _m(*keys):
        for k in keys:
            if k in rd:
                return round(float(rd[k]), 4)
        return None

    metrics = {
        "mAP50": _m("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50_95": _m("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "precision": _m("metrics/precision(B)", "metrics/precision"),
        "recall": _m("metrics/recall(B)", "metrics/recall"),
    }
    return {"best": best, "metrics": metrics, "epochs": epochs}


def main() -> int:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_FILE.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    model_key = manifest["model_key"]      # custom/<user>/<job>/best.pt
    result_key = manifest["result_key"]    # training/<job>/result.json

    try:
        ds_dir = WORK / "dataset"
        data_yaml = _build_dataset(manifest, ds_dir)
        out = _train(manifest, data_yaml)

        # SageMaker artifact (model.tar.gz) + an explicit copy to the models
        # bucket so the inference ModelManager can load it by key.
        import shutil
        shutil.copy(str(out["best"]), str(MODEL_DIR / "best.pt"))
        _s3.upload_file(str(out["best"]), MODELS_BUCKET, model_key)
        log(f"uploaded best.pt -> s3://{MODELS_BUCKET}/{model_key}")

        result = {
            "status": "completed",
            "storage_key": model_key,
            "metrics": out["metrics"],
            "epochs_completed": out["epochs"],
            "execution_seconds": round(time.time() - t0, 1),
        }
        _s3.put_object(
            Bucket=OUTPUT_BUCKET, Key=result_key,
            Body=json.dumps(result).encode("utf-8"), ContentType="application/json",
        )
        log(f"done in {result['execution_seconds']}s; metrics={out['metrics']}")
        return 0
    except Exception as e:  # noqa: BLE001 - surface to SageMaker FailureReason
        err = f"{e}\n{traceback.format_exc()}"
        log(f"FAILED: {err}")
        FAILURE_FILE.write_text(str(e)[:1024])
        try:
            _s3.put_object(
                Bucket=OUTPUT_BUCKET, Key=result_key,
                Body=json.dumps({"status": "failed", "error": str(e)[:2000]}).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
