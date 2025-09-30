# Backend

## Quick start

1. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   cd backend
   uv sync
   ```
2. Launch the FastAPI server:
   ```bash
   uv run uvicorn backend.app.main:app --reload
   ```
3. Submit a job via the new API:
   ```bash
   curl -X POST http://localhost:8000/animate/jobs \
     -H 'Content-Type: application/json' \
     -d '{"src_root_path": "data/jobs/demo"}'
   ```
   Poll `GET /animate/jobs/{job_id}` until it reports `succeeded`, then download the result from `/animate/jobs/{job_id}/media`.

## Command-line inference

The same pipeline is exposed through `backend/app/infer.py` and the helper script `scripts/torchrun_animate.sh`.

- Single GPU:
  ```bash
  uv run python -m backend.app.infer --src-root data/jobs/demo
  ```
- Two GPUs with sequence parallelism:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    backend/app/infer.py --src-root data/jobs/demo
  ```
  Sequence parallel is enabled automatically when `WORLD_SIZE > 1`. Set `WAN_ENABLE_FSDP=1` to activate FSDP sharding as well.

The CLI prints a JSON payload containing the final status (job id, video path, timestamps). Use `--help` to review all tunable parameters.

## Model weights

By default the service downloads Wan2.2 Animate 14B from ModelScope on first use. To reuse a pre-downloaded snapshot, export one of the following before starting the backend:

- `WAN_CHECKPOINT_DIR=/path/to/checkpoint`
- `WAN_MODELS_ROOT=/path/to/cache` (will hold the ModelScope snapshot)

Other useful knobs:

- `WAN_MODEL_REVISION` — pin to a specific ModelScope revision.
- `WAN_DEFAULT_OFFLOAD_MODEL=false` — keep the text encoder on GPU between runs.
- `WAN_ENABLE_SEQUENCE_PARALLEL=false` — force single-GPU execution even when launched with `torchrun`.

## Smoke tests

After syncing dependencies you can perform a dry run to verify the pipeline wiring:

```bash
# Prepare a minimal job directory that already contains src_pose.mp4/src_face.mp4/src_ref.png
uv run python -m backend.app.infer --src-root data/jobs/demo --sampling-steps 2 --clip-len 5
```

The command finishes quickly (low-quality output) and confirms that asset discovery, model loading, and result serialization all work end-to-end.
