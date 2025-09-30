from fastapi import FastAPI

app = FastAPI(title="Wan Local Suite API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
