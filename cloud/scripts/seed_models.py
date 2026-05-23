#!/usr/bin/env python3
"""Upload local model files to the S3 models bucket.

Usage:
    python -m cloud.scripts.seed_models
    python -m cloud.scripts.seed_models --models-dir /path/to/models
"""

import argparse
import logging
from pathlib import Path

from cloud.storage.s3_client import S3StorageClient
from cloud.storage.config import S3Config
from cloud.wrapper.model_manager import ModelManager, MODEL_FILES

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main():
    parser = argparse.ArgumentParser(description="Seed BeeMonitor models to S3")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing model files",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify uploads by comparing sizes"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error("Models directory not found: %s", models_dir)
        return

    # Check which files exist locally
    print(f"Models directory: {models_dir}")
    for key, filename in MODEL_FILES.items():
        path = models_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {key}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  {key}: {filename} (NOT FOUND)")

    # Upload
    config = S3Config()
    client = S3StorageClient(config)
    manager = ModelManager(storage_client=client, storage_config=config)

    print("\nUploading models to S3...")
    uploaded = manager.upload_models(str(models_dir))
    for key, blob_path in uploaded.items():
        print(f"  Uploaded: {key} -> {blob_path}")

    if not uploaded:
        print("  No models uploaded (files missing).")
        return

    # Verify
    if args.verify:
        print("\nVerifying uploads...")
        results = manager.verify_models(str(models_dir))
        for key, ok in results.items():
            status = "OK" if ok else "MISMATCH"
            print(f"  {key}: {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
