from __future__ import annotations

from fastapi import FastAPI

from .routers import animate

app = FastAPI(title="Wan Local Suite API")
app.include_router(animate.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
