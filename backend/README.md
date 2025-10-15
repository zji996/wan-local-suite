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

By default the service downloads Wan2.2 Animate 14B from ModelScope on first use. You can pre-fetch the weights with the helper script:

```bash
uv run python scripts/download_wan_model.py
```

To reuse a pre-downloaded snapshot, export one of the following before starting the backend:

- `WAN_CHECKPOINT_DIR=/path/to/checkpoint`
- `WAN_MODELS_ROOT=/path/to/cache` (will hold the ModelScope snapshot)

Other useful knobs:

- `WAN_MODEL_REVISION` — pin to a specific ModelScope revision.
- `WAN_DEFAULT_OFFLOAD_MODEL=false` — keep the text encoder on GPU between runs.
- `WAN_ENABLE_SEQUENCE_PARALLEL=false` — force single-GPU execution even when launched with `torchrun`.

## DFloat11 compression (fits dual RTX 3080 20 GB)

The backend can now load Wan2.2 Animate in the DF11 lossless format, shrinking the DiT weights by ~30% with identical output quality. This is ideal for 20 GB GPUs such as dual RTX 3080s.

1. Ensure dependencies are synced so `dfloat11` is available:
   ```bash
   cd backend
   uv sync
   ```
2. Compress the original Wan2.2 checkpoint:
   ```bash
   uv run python scripts/compress_wan22_df11.py \
     --checkpoint-dir models/Wan2.2-Animate-14B \
     --output-dir models/Wan2.2-Animate-14B-DF11 \
     --save-single-file
   ```
   The script packs the DiT blocks, face adapter, and head into DF11 tensors and writes a new safetensors checkpoint.
3. Point the backend to the compressed weights by setting:
   ```bash
   export WAN_USE_DFLOAT11=true
   export WAN_DFLOAT11_LOCAL_DIR=models/Wan2.2-Animate-14B-DF11
   export WAN_DFLOAT11_MAX_MEMORY='{"0":"19GiB","1":"19GiB"}'
   ```
   Optional toggles:
   - `WAN_DFLOAT11_CPU_OFFLOAD=true` keeps only one block on GPU (useful for 16 GB cards).

**Limitations:** DF11 is incompatible with `WAN_ENABLE_SEQUENCE_PARALLEL` and `WAN_ENABLE_FSDP`; both are automatically disabled when DF11 is active.

### Launching the backend on two GPUs (single process)

Once the DF11 weights are in place, run the FastAPI server directly from the repo root:

```bash
cd /home/zji/wan-local-suite
export WAN_USE_DFLOAT11=true
export WAN_DFLOAT11_LOCAL_DIR=models/Wan2.2-Animate-14B-DF11
export WAN_DFLOAT11_MAX_MEMORY='{"0":"19GiB","1":"19GiB"}'
export WAN_ENABLE_SEQUENCE_PARALLEL=false
export WAN_ENABLE_FSDP=false
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
  uv run --project backend \
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

`uv run --project backend` ensures uv uses the existing `backend/.venv` environment. Adjust `WAN_DFLOAT11_MAX_MEMORY` to match your cards; set `WAN_DFLOAT11_CPU_OFFLOAD=true` if 20 GB per GPU is still tight (expect some slowdown).

## Smoke tests

After syncing dependencies you can perform a dry run to verify the pipeline wiring:

```bash
# Prepare a minimal job directory that already contains src_pose.mp4/src_face.mp4/src_ref.png
uv run python -m backend.app.infer --src-root data/jobs/demo --sampling-steps 2 --clip-len 5
```

The command finishes quickly (low-quality output) and confirms that asset discovery, model loading, and result serialization all work end-to-end.

## Multi-GPU serving

When you have multiple GPUs available you can launch the FastAPI service under `torchrun` so the `WanAnimate` model shreds across devices. Use the helper script from the repo root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WAN_CHECKPOINT_DIR=models/Wan2.2-Animate-14B \
WAN_ENABLE_SEQUENCE_PARALLEL=true \
WAN_ENABLE_FSDP=true \
WAN_USE_RELIGHTING_LORA=true \
PYTHONPATH=. \
backend/.venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/serve_multigpu.py --host 0.0.0.0 --port 8000
```

Rank 0 serves the API while the remaining ranks stay alive for distributed inference. Adjust the environment variables (`WAN_*`) to match your local setup.

> The backend now preloads WanAnimate on startup, so the first request no longer pays the model load penalty.
