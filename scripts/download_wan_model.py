#!/usr/bin/env python3
"""Download Wan2.2 Animate checkpoints from ModelScope into the local models cache."""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-Animate-14B"
DEFAULT_OUTPUT = Path("models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"ModelScope model identifier (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--revision",
        help="Optional ModelScope revision or branch to download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to store the downloaded checkpoint (default: models/).",
    )
    parser.add_argument(
        "--local-subdir",
        help="Optional name for the model folder under the output directory (defaults to model-id suffix).",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Disable symlink usage when snapshot_download creates the local directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = output_dir / (args.local_subdir or args.model_id.split("/")[-1])
    local_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading %s to %s", args.model_id, local_dir)

    kwargs = dict(
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=str(output_dir),
        local_dir=str(local_dir),
    )

    signature = inspect.signature(snapshot_download)
    if "local_dir_use_symlinks" in signature.parameters:
        kwargs["local_dir_use_symlinks"] = not args.no_symlinks
    elif args.no_symlinks:
        logging.warning("Current ModelScope version does not support disabling symlinks; ignoring --no-symlinks")

    snapshot_path = snapshot_download(**kwargs)

    logging.info("Model snapshot ready at %s", snapshot_path)


if __name__ == "__main__":
    main()
