import os
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "banksight/"

# what each "set" means: a list of folders / files relative to the project root
SETS = {
    "reports": ["reports"],
    "models": ["models"],
    "warehouse": ["data/warehouse/banksight.duckdb"],
}


def get_client():
    """Build an S3 client from the AWS CLI profile / env (no keys in code)."""
    return boto3.client("s3")


def get_bucket():
    """Read the bucket name from the environment."""
    bucket = os.environ.get("BANKSIGHT_S3_BUCKET")
    if not bucket:
        sys.exit("Set BANKSIGHT_S3_BUCKET first, e.g.  set BANKSIGHT_S3_BUCKET=my-banksight-bucket")
    return bucket


def local_files(set_name):
    """List every local file that belongs to a set."""
    files = []
    for item in SETS[set_name]:
        path = ROOT / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return files


def remote_sizes(s3, bucket, prefix):
    """Return {key: size} for everything under a prefix."""
    sizes = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def upload(set_name, s3=None, bucket=None):
    """Upload a set; skip files whose size already matches the remote copy."""
    s3 = s3 or get_client()
    bucket = bucket or get_bucket()
    existing = remote_sizes(s3, bucket, PREFIX)
    sent = skipped = 0
    for path in local_files(set_name):
        key = PREFIX + path.relative_to(ROOT).as_posix()
        if existing.get(key) == path.stat().st_size:
            skipped += 1
            continue
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
        print("uploaded", key)
        sent += 1
    print(f"done: {sent} uploaded, {skipped} unchanged")
    return sent, skipped


def download(set_name, s3=None, bucket=None):
    """Download every remote file that belongs to a set into the project folder."""
    s3 = s3 or get_client()
    bucket = bucket or get_bucket()
    wanted = tuple(PREFIX + Path(i).as_posix() for i in SETS[set_name])
    got = 0
    for key in remote_sizes(s3, bucket, PREFIX):
        if not key.startswith(wanted):
            continue
        target = ROOT / key[len(PREFIX):]
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        print("downloaded", key)
        got += 1
    print(f"done: {got} downloaded")
    return got


def show_list(s3=None, bucket=None):
    """Print what is in the bucket."""
    s3 = s3 or get_client()
    bucket = bucket or get_bucket()
    total = 0
    for key, size in remote_sizes(s3, bucket, PREFIX).items():
        print(f"{size/1e6:9.2f} MB  {key}")
        total += size
    print(f"total {total/1e6:.1f} MB")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["list"]:
        show_list()
    elif len(args) == 2 and args[0] in ("upload", "download") and args[1] in SETS:
        (upload if args[0] == "upload" else download)(args[1])
    else:
        sys.exit(__doc__)
