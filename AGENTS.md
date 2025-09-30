# Wan Local Suite - Agent Notes

- Use `make dev` to bootstrap both backend and frontend while developing.
- Scripts in `scripts/` are the canonical entry points for preprocessing and inference dispatch.
- The FastAPI service lives in `backend/` and should expose health checks at `/health`.
- Frontend assets are maintained under `frontend/src/` with React and styled-components as the styling layer.
- Third-party dependencies such as Wan2.2 reside in `third_party/` and should be managed as git submodules.
