#!/usr/bin/env python3
"""Run inside Kaggle: download Drive folder, upload/version Kaggle dataset."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DRIVE_FOLDER = "__DRIVE_FOLDER__"
DATASET_ID = "__DATASET_ID__"
DATASET_TITLE = "__DATASET_TITLE__"
DATASET_SUBTITLE = "__DATASET_SUBTITLE__"
DATASET_DESCRIPTION = "__DATASET_DESCRIPTION__"
DATASET_LICENSE = "__DATASET_LICENSE__"
UPLOAD_MODE = "__UPLOAD_MODE__"
DIR_MODE = "__DIR_MODE__"
PUBLIC = __PUBLIC_BOOL__
VERSION_MESSAGE = "__VERSION_MESSAGE__"
STAGING = Path("/kaggle/working") / (
    "drive_dataset_staging_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
)


def run(command, check=True):
    print("+ " + " ".join(map(str, command)))
    return subprocess.run(command, check=check, text=True)


def ensure_package(package):
    try:
        __import__(package)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", package])


def secret(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def configure_kaggle_auth():
    username = secret("KAGGLE_USERNAME")
    key = secret("KAGGLE_KEY") or secret("KAGGLE_API_TOKEN")
    if not username or not key:
        raise SystemExit(
            "Missing Kaggle credentials. Add Kaggle notebook secrets "
            "`KAGGLE_USERNAME` and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`)."
        )
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def write_metadata():
    metadata = {
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [{"name": DATASET_LICENSE}],
        "subtitle": DATASET_SUBTITLE,
        "description": DATASET_DESCRIPTION,
    }
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def payload_count():
    return sum(
        1
        for path in STAGING.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    )


def dataset_exists():
    result = subprocess.run(
        ["kaggle", "datasets", "files", DATASET_ID],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def upload_dataset():
    mode = UPLOAD_MODE
    if mode == "auto":
        mode = "version" if dataset_exists() else "create"

    if mode == "create":
        command = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(STAGING),
            "--dir-mode",
            DIR_MODE,
        ]
        if PUBLIC:
            command.append("--public")
    else:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(STAGING),
            "-m",
            VERSION_MESSAGE,
            "--dir-mode",
            DIR_MODE,
        ]
    run(command)


def main():
    ensure_package("gdown")
    configure_kaggle_auth()

    STAGING.mkdir(parents=True, exist_ok=True)
    run(["gdown", "--folder", DRIVE_FOLDER, "-O", str(STAGING), "--remaining-ok"])
    write_metadata()

    count = payload_count()
    print(f"Payload files: {count}")
    if count == 0:
        raise SystemExit("No files downloaded. Check Drive folder sharing/access.")

    upload_dataset()
    print(f"Done: {DATASET_ID}")


if __name__ == "__main__":
    main()
