"""Revive raw clips stuck in Glacier Flexible: `restore` asks for a Bulk
restore of each; `copy` copies each restored one in place to GLACIER_IR;
`status` counts where they stand. Keys are listed fresh each run."""
import sys, json, boto3
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = "beemonitor-dev-raw-videos-495331821764"
s3 = boto3.Session(profile_name="ecomorph").client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 10, "mode": "adaptive"}))

def glacier_keys():
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for o in page.get("Contents", []):
            if o["StorageClass"] == "GLACIER":
                yield o["Key"]

def restore(key):
    try:
        s3.restore_object(Bucket=BUCKET, Key=key, RestoreRequest={"Days": 10, "GlacierJobParameters": {"Tier": "Bulk"}})
        return "requested"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        return {"RestoreAlreadyInProgress": "in progress"}.get(code, "error " + code)

def state(key):
    r = s3.head_object(Bucket=BUCKET, Key=key).get("Restore") or ""
    return "restored" if 'ongoing-request="false"' in r else ("in progress" if r else "not requested")

def copy(key):
    st = state(key)
    if st != "restored":
        return st
    try:
        s3.copy_object(Bucket=BUCKET, Key=key, CopySource={"Bucket": BUCKET, "Key": key},
                       StorageClass="GLACIER_IR", MetadataDirective="COPY")
        return "copied"
    except ClientError as e:
        return "error " + e.response["Error"]["Code"]

def run(fn, keys):
    counts, errors = {}, []
    with ThreadPoolExecutor(32) as pool:
        for k, r in zip(keys, pool.map(fn, keys)):
            counts[r] = counts.get(r, 0) + 1
            if r.startswith("error"):
                errors.append((k, r))
    print(json.dumps(counts, indent=1)); print("errors sample:", errors[:5])

if __name__ == "__main__":
    keys = list(glacier_keys())
    print(len(keys), "objects still in GLACIER")
    run({"restore": restore, "copy": copy, "status": state}[sys.argv[1]], keys)
