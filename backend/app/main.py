from __future__ import annotations

from fastapi import FastAPI

from .routers import animate
from .routers.animate import service as animate_service

app = FastAPI(title="Wan Local Suite API")
app.include_router(animate.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def _preload_wan_animate() -> None:
    animate_service.preload_model()
