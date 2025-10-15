#!/usr/bin/env python3
"""Launch the FastAPI backend under torchrun, serving on rank 0 only."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import uvicorn

# Importing the application sets up AnimateService on every rank.
try:
    from backend.app.main import app  # type: ignore[attr-defined]
    from backend.app.routers.animate import service as animate_service
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for misconfigured PYTHONPATH
    raise SystemExit(
        "Failed to import backend.app.*. Make sure you run this script from the repo root"
    ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interface for Uvicorn to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for Uvicorn to bind (default: 8000)")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level (default: info)")
    parser.add_argument(
        "--idle-log-seconds",
        type=int,
        default=300,
        help="Log a heartbeat message this often on non-serving ranks (default: 300)",
    )
    return parser.parse_args(argv)


def serve_rank_zero(host: str, port: int, log_level: str) -> None:
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    server.run()


def idle_rank(rank: int, interval: int) -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Rank %s entering idle loop; waiting for shutdown signal", rank)

    def _handle_sigterm(signum: int, frame) -> None:  # pragma: no cover - signal handler
        logger.info("Rank %s received signal %s; exiting", rank, signum)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        while True:
            time.sleep(interval)
            logger.info("Rank %s still alive", rank)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.exception("Idle loop on rank %s terminated unexpectedly", rank)
        raise SystemExit(1) from exc


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    logging.basicConfig(level=logging.INFO)

    try:
        animate_service.preload_model()
    except Exception as exc:  # pragma: no cover - surface load failures early
        raise SystemExit(f"Rank {rank} failed to preload WanAnimate: {exc}") from exc

    if world_size <= 1 or rank == 0:
        serve_rank_zero(args.host, args.port, args.log_level)
        return

    idle_rank(rank, args.idle_log_seconds)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
