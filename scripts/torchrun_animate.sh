#!/usr/bin/env bash
set -euo pipefail

# Placeholder script for launching distributed animation inference with torchrun.
# Example: ./scripts/torchrun_animate.sh --ckpt ./models/latest --config ./configs/infer.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  "$ROOT_DIR/backend/app/infer.py" "$@"
