# Repository Guidelines

## Project Structure & Module Organization
- `backend/` hosts the FastAPI service; `backend/app/main.py` exposes `/health`, and additional routers should live under `backend/app/`.
- `frontend/` contains the React + TypeScript UI with styled-components; principal code resides in `frontend/src/`, supporting assets in `frontend/public/`.
- `scripts/` provides canonical CLIs (`preprocess.sh`, `torchrun_animate.sh`) for data prep and inference; update these before introducing new entry points.
- `data/` collects local inputs and artifacts, while `models/` is reserved for checkpoints kept out of version control.
- `third_party/` tracks vendored submodules such as `wan2.2`; manage them strictly through git submodule commands.

## Build, Test, and Development Commands
- `make dev` launches backend (`uv run --python backend/app/main.py`) and frontend (`npm run dev`) together.
- `make backend` or `make frontend` runs a single stack; export any required env vars first.
- `make preprocess job=path/to/job.yaml` writes processed assets into `data/jobs/`.
- `make infer args="--ckpt ./models/latest --config ./configs/infer.yaml"` triggers `torchrun`-based inference.
- After dependency edits run `uv sync` inside `backend/` and `npm install` (or `pnpm install`) inside `frontend/`.

## DFloat11 Workflow
- Compress Wan2.2 Animate checkpoints with `uv run --project backend python scripts/compress_wan22_df11.py --checkpoint-dir models/Wan2.2-Animate-14B --output-dir models/Wan2.2-Animate-14B-DF11 --save-single-file`.
- When using DF11 set the env vars: `WAN_USE_DFLOAT11=true`, `WAN_DFLOAT11_LOCAL_DIR=models/Wan2.2-Animate-14B-DF11`, and cap GPU memory with `WAN_DFLOAT11_MAX_MEMORY='{"0":"19GiB","1":"19GiB"}'`; optionally enable `WAN_DFLOAT11_CPU_OFFLOAD=true` for tighter VRAM budgets.
- Launch the backend on dual GPUs via a single process:  
  `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 uv run --project backend uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`. DF11 disables sequence parallelism and FSDP automatically.
- Keep T5/CLIP/VAE weights in their original formats; only the diffusion transformer requires DF11 compression.

## Coding Style & Naming Conventions
- Python follows PEP 8 with four-space indents and type hints on public functions; colocate domain modules under `backend/app/`.
- Format Python with `black` and lint with `ruff` (`uv tool install black ruff`) before submitting.
- React components live in `frontend/src/` using `PascalCase.tsx`; hooks/utilities use `camelCase.ts`.
- Keep styled-components next to their owners; elevate shared tokens into a small theme module when duplication appears.

## Testing Guidelines
- Add backend tests under `backend/tests/` with `pytest`; run via `uv run pytest` and target ≥80% coverage on touched modules.
- For the frontend, wire up `vitest` + React Testing Library in `frontend/src/__tests__/` and execute with `npm test` once scripts are registered.
- Document manual smoke steps for shell script updates until automated coverage is available.
- Note any gaps or skipped cases in the pull request checklist.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., `feat: add storyboard preview panel`) and expand in the body when altering schemas or filesystems.
- Rebase before opening PRs; include screenshots or CLI output for UI, data, or pipeline changes.
- Reference issue IDs and list the validation commands you ran (`make dev`, `uv run pytest`, `npm run lint`, etc.).
- Update docs—including this guide—whenever workflows change, and request reviews from both backend and frontend owners for cross-stack work.
