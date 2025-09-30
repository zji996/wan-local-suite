#!/usr/bin/env bash
set -euo pipefail

# Placeholder for data preprocessing pipeline.
# Example: ./scripts/preprocess.sh data/jobs/job.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_CONFIG="${1:-}"

if [[ -z "$JOB_CONFIG" ]]; then
  echo "Usage: $0 <job-config>" >&2
  exit 1
fi

python -m preprocess --config "$JOB_CONFIG" --output "$ROOT_DIR/data/jobs"
