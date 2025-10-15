#!/usr/bin/env python3
"""Compress Wan 2.2 Animate checkpoints into the DF11 format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from dfloat11 import compress_model
from third_party.wan2_2.wan.dfloat11_pattern import get_wan22_animate_pattern_dict
from third_party.wan2_2.wan.modules.animate import WanAnimateModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the original Wan2.2 Animate BFloat16 checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the DF11 compressed checkpoint will be written.",
    )
    parser.add_argument(
        "--block-range",
        type=int,
        nargs=2,
        default=(0, 10000),
        metavar=("START", "END"),
        help="Optional block index range for parallel compression (default: 0 10000).",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Run a GPU round-trip check for every block (adds time but verifies decompression).",
    )
    parser.add_argument(
        "--save-single-file",
        action="store_true",
        help="Store the compressed weights in a single safetensors file (default: multi-file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern_dict = get_wan22_animate_pattern_dict()

    logging.info("Loading WanAnimateModel (bf16) from %s", checkpoint_dir)
    model = WanAnimateModel.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    logging.info(
        "Compressing WanAnimateModel into DF11 format (output=%s, single_file=%s, correctness=%s)",
        output_dir,
        args.save_single_file,
        args.check_correctness,
    )
    compress_model(
        model=model,
        pattern_dict=pattern_dict,
        save_path=str(output_dir),
        block_range=args.block_range,
        save_single_file=args.save_single_file,
        check_correctness=args.check_correctness,
    )
    logging.info("Compression complete. DF11 checkpoint saved to %s", output_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
