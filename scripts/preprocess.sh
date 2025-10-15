#!/usr/bin/env bash
set -euo pipefail

# Launch the Wan-Animate preprocessing pipeline using a YAML job specification.
# Example: ./scripts/preprocess.sh examples/preprocess_demo/replace_job.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_CONFIG="${1:-}"

if [[ -z "$JOB_CONFIG" ]]; then
  echo "Usage: $0 <job-config>" >&2
  exit 1
fi

shift

if command -v realpath >/dev/null 2>&1; then
  JOB_CONFIG_ABS="$(realpath "$JOB_CONFIG")"
else
  JOB_CONFIG_ABS="$(JOB_CONFIG="$JOB_CONFIG" python3 - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["JOB_CONFIG"]).expanduser().resolve()
print(path)
PY
)"
fi

if [[ ! -f "$JOB_CONFIG_ABS" ]]; then
  echo "Job config not found: $JOB_CONFIG_ABS" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run python)
else
  PYTHON_RUNNER=(python3)
fi

cd "$ROOT_DIR"

"${PYTHON_RUNNER[@]}" "$ROOT_DIR/scripts/preprocess_job.py" --config "$JOB_CONFIG_ABS" "$@"
