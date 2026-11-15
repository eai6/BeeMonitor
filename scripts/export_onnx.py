"""Give YOLO models in the models bucket their CPU copy (memory/82).

The recount worker finds each clip's tubes with OpenCV on CPU, from an ONNX
copy of the model beside its weights: ``x/best.pt`` -> ``x/best.onnx`` and
``x/best.onnx.json`` (its class names and input size). Training writes them
for new models (sagemaker_backend/train.py); this writes them for the
built-in models and models trained before.

Run on a machine with ultralytics (the repo's venv) and AWS access:

    pip install onnx onnxslim               # once, for the export
    AWS_PROFILE=ecomorph python scripts/export_onnx.py --bucket <models bucket>
    AWS_PROFILE=ecomorph python scripts/export_onnx.py --bucket <models bucket> custom/7/42/best.pt

With no keys: the built-in models (v1/nest_detection.pt, v1/bee_tracking.pt)
and every ``custom/.../*.pt`` that has no copy yet. ``--dry-run`` lists them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

BUILTIN = ("v1/nest_detection.pt", "v1/bee_tracking.pt")


def base(key: str) -> str:
    return key[:-3] if key.endswith(".pt") else key


def todo(s3, bucket):
    """The built-in models, then every custom model without its copy."""
    keys, have = [], set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="custom/"):
        for obj in page.get("Contents") or []:
            k = obj["Key"]
            if k.endswith(".pt"):
                keys.append(k)
            elif k.endswith(".onnx.json"):
                have.add(k[: -len(".onnx.json")])
    return [k for k in BUILTIN if _exists(s3, bucket, k)] + [k for k in keys if base(k) not in have]


def _exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001
        return False


def export(s3, bucket, key, folder: Path):
    from ultralytics import YOLO

    local = folder / (base(key).replace("/", "_") + ".pt")
    s3.download_file(bucket, key, str(local))
    model = YOLO(str(local))
    if getattr(model, "task", "detect") != "detect":
        print(f"  skipped {key}: a {model.task} model, not a detector")
        return
    names = [model.names[i] for i in sorted(model.names)]
    imgsz = (getattr(model.model, "args", None) or {}).get("imgsz") or 640
    imgsz = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
    path = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
    s3.upload_file(str(path), bucket, base(key) + ".onnx")
    s3.put_object(Bucket=bucket, Key=base(key) + ".onnx.json", ContentType="application/json",
                  Body=json.dumps({"names": names, "imgsz": imgsz, "source": key}).encode())
    print(f"  {key} -> {base(key)}.onnx  classes: {', '.join(names)}  input {imgsz}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bucket", required=True, help="the models bucket, e.g. beemonitor-dev-models-<account>")
    ap.add_argument("--dry-run", action="store_true", help="list what would be exported")
    ap.add_argument("keys", nargs="*", help="model keys in the bucket (default: all without a copy)")
    args = ap.parse_args(argv)

    import boto3

    s3 = boto3.client("s3")
    keys = args.keys or todo(s3, args.bucket)
    print(f"{len(keys)} model(s) to export")
    if args.dry_run:
        print("\n".join(f"  {k}" for k in keys))
        return 0
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for key in keys:
            try:
                export(s3, args.bucket, key, Path(tmp))
            except Exception as exc:  # noqa: BLE001 - one bad model doesn't stop the rest
                failed += 1
                print(f"  FAILED {key}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
