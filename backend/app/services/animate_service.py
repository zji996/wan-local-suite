from __future__ import annotations

import importlib
import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from imageio import v2 as imageio

from third_party.wan2.2.wan.animate import WanAnimate

from ..config import AnimateSettings, get_settings
from ..schemas.animate import AnimateJobCreate, AnimateJobStatus

logger = logging.getLogger(__name__)


class AnimateService:
    def __init__(self, settings: Optional[AnimateSettings] = None) -> None:
        self.settings = settings or get_settings()
        self.jobs_root = self.settings.jobs_root.expanduser().resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.rank, self.world_size = self._setup_distributed()
        self.device_id = int(os.environ.get("LOCAL_RANK", self.settings.device_id))

        self._model_lock = Lock()
        self._model: Optional[WanAnimate] = None
        self._config_template = None
        self._checkpoint_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_job(self, request: AnimateJobCreate, allow_existing: bool = False) -> AnimateJobStatus:
        job_id = request.job_id or self._generate_job_id()
        job_dir = self.jobs_root / job_id

        if job_dir.exists():
            if not allow_existing:
                raise FileExistsError(f"Job directory already exists: {job_dir}")
        else:
            job_dir.mkdir(parents=True, exist_ok=False)

        if self.rank == 0:
            self._write_request(job_dir, request)
            status = AnimateJobStatus(job_id=job_id, status="pending")
            self._write_status(job_dir, status)
        else:
            status = AnimateJobStatus(job_id=job_id, status="pending")

        return status

    def get_status(self, job_id: str) -> AnimateJobStatus:
        job_dir = self.jobs_root / job_id
        status_path = job_dir / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(f"Status file not found for job {job_id}")

        data = json.loads(status_path.read_text(encoding="utf-8"))
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("finished_at"):
            data["finished_at"] = datetime.fromisoformat(data["finished_at"])
        if data.get("output_path"):
            data["output_path"] = Path(data["output_path"]).expanduser().resolve()
        return AnimateJobStatus(**data)

    def execute_job(self, job_id: str, request: AnimateJobCreate) -> None:
        self._execute_job(job_id, request)

    def run_job_sync(self, request: AnimateJobCreate) -> AnimateJobStatus:
        if self.world_size > 1:
            job_id = request.job_id or ""
            payload = None
            if self.rank == 0:
                status = self.create_job(request, allow_existing=bool(request.job_id))
                job_id = status.job_id
                payload = request.model_dump(mode="json")
            job_id = self._broadcast_string(job_id)
            payload = self._broadcast_payload(payload)
            request = AnimateJobCreate(**payload)
            self._ensure_job_dir(job_id)
            self._execute_job(job_id, request)
            self._barrier()
            if self.rank == 0:
                return self.get_status(job_id)
            return AnimateJobStatus(job_id=job_id, status="succeeded")

        status = self.create_job(request, allow_existing=bool(request.job_id))
        job_id = status.job_id
        self._execute_job(job_id, request)
        return self.get_status(job_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_job(self, job_id: str, request: AnimateJobCreate) -> None:
        job_dir = self.jobs_root / job_id
        started_at: Optional[datetime] = None
        if self.rank == 0:
            started_at = datetime.utcnow()
            running = AnimateJobStatus(job_id=job_id, status="running", started_at=started_at)
            self._write_status(job_dir, running)

        self._barrier()

        try:
            self._validate_source_assets(request.src_root_path, request.replace_flag)
            model = self._load_model()
            generate_kwargs = self._build_generate_kwargs(request)
            video_tensor = model.generate(**generate_kwargs)

            output_path: Optional[Path] = None
            if self.rank == 0 and video_tensor is not None:
                output_path = job_dir / "output.mp4"
                fps = request.fps or getattr(self._load_config_template(), "sample_fps", self.settings.default_fps)
                self._write_video(video_tensor, output_path, fps=fps)

            self._barrier()

            if self.rank == 0:
                succeeded = AnimateJobStatus(
                    job_id=job_id,
                    status="succeeded",
                    output_path=output_path,
                    started_at=started_at,
                    finished_at=datetime.utcnow(),
                )
                self._write_status(job_dir, succeeded)
        except Exception as exc:  # noqa: BLE001
            self._barrier()
            if self.rank == 0:
                failed = AnimateJobStatus(
                    job_id=job_id,
                    status="failed",
                    message=str(exc),
                    started_at=started_at,
                    finished_at=datetime.utcnow(),
                )
                self._write_status(job_dir, failed)
            logger.exception("Job %s failed", job_id)
            raise

    def _load_model(self) -> WanAnimate:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            if not torch.cuda.is_available():
                raise RuntimeError("WanAnimate requires at least one CUDA device")

            checkpoint_dir = self._resolve_checkpoint_dir()
            config = deepcopy(self._load_config_template())

            use_sp = self.settings.enable_sequence_parallel
            if use_sp is None:
                use_sp = self.world_size > 1

            if self.world_size > 1 and not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized for multi-GPU inference")

            logger.info(
                "Loading WanAnimate model from %s (device cuda:%s, sp=%s, fsdp=%s)",
                checkpoint_dir,
                self.device_id,
                use_sp,
                self.settings.enable_fsdp,
            )

            model = WanAnimate(
                config=config,
                checkpoint_dir=str(checkpoint_dir),
                device_id=self.device_id,
                rank=self.rank,
                use_sp=use_sp,
                dit_fsdp=self.settings.enable_fsdp,
                init_on_cpu=self.settings.init_on_cpu,
                use_relighting_lora=self.settings.use_relighting_lora,
            )

            self._model = model
            return model

    def _load_config_template(self):
        if self._config_template is None:
            module = importlib.import_module(self.settings.config_module)
            template = getattr(module, self.settings.config_attr)
            self._config_template = deepcopy(template)
        return self._config_template

    def _resolve_checkpoint_dir(self) -> Path:
        if self._checkpoint_dir is not None:
            return self._checkpoint_dir

        if self.settings.checkpoint_dir:
            path = self.settings.checkpoint_dir.expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {path}")
            self._checkpoint_dir = path
            return path

        models_root = self.settings.models_root.expanduser().resolve()
        models_root.mkdir(parents=True, exist_ok=True)

        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:  # noqa: F841
            raise RuntimeError(
                "ModelScope is required to download WAN 2.2 checkpoints. Install with `uv add modelscope`."
            ) from exc

        logger.info(
            "Downloading model weights from ModelScope (%s) into %s",
            self.settings.model_id,
            models_root,
        )
        local_dir = snapshot_download(
            self.settings.model_id,
            cache_dir=str(models_root),
            revision=self.settings.model_revision,
        )
        path = Path(local_dir).expanduser().resolve()
        self._checkpoint_dir = path
        return path

    def _build_generate_kwargs(self, request: AnimateJobCreate) -> Dict[str, Any]:
        config = self._load_config_template()
        clip_len = request.clip_len or getattr(config, "frame_num", self.settings.default_clip_len)
        refert_num = request.refert_num or self.settings.default_refert_num
        shift = request.shift or getattr(config, "sample_shift", self.settings.default_shift)
        sample_solver = request.sample_solver or self.settings.default_sample_solver
        sampling_steps = request.sampling_steps or getattr(config, "sample_steps", self.settings.default_sampling_steps)
        guide_scale = request.guide_scale or getattr(config, "sample_guide_scale", self.settings.default_guide_scale)
        seed = request.seed if request.seed is not None else self.settings.default_seed
        offload_model = (
            request.offload_model if request.offload_model is not None else self.settings.default_offload_model
        )

        if clip_len % 4 != 1:
            raise ValueError("clip_len must satisfy 4n + 1")
        if refert_num not in (1, 5):
            raise ValueError("refert_num must be 1 or 5")

        return {
            "src_root_path": str(request.src_root_path),
            "replace_flag": request.replace_flag,
            "clip_len": clip_len,
            "refert_num": refert_num,
            "shift": shift,
            "sample_solver": sample_solver,
            "sampling_steps": sampling_steps,
            "guide_scale": guide_scale,
            "input_prompt": request.input_prompt or "",
            "n_prompt": request.negative_prompt or "",
            "seed": seed,
            "offload_model": offload_model,
        }

    def _write_video(self, tensor: torch.Tensor, output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
        frames = np.clip((frames + 1.0) * 0.5, 0.0, 1.0)
        frames = (frames * 255).astype(np.uint8)

        with imageio.get_writer(output_path, format="mp4", fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

        logger.info("Saved generated video to %s", output_path)

    def _write_request(self, job_dir: Path, request: AnimateJobCreate) -> None:
        payload = request.model_dump(mode="json", exclude_none=True)
        (job_dir / "request.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_status(self, job_dir: Path, status: AnimateJobStatus) -> None:
        payload = status.model_dump(mode="json", exclude_none=True)
        (job_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _validate_source_assets(self, src_root: Path, replace_flag: bool) -> None:
        required = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
        if replace_flag:
            required += ["src_bg.mp4", "src_mask.mp4"]

        missing = [name for name in required if not (src_root / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required source files in {src_root}: {', '.join(missing)}"
            )

    def _ensure_job_dir(self, job_id: str) -> None:
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

    def _broadcast_string(self, value: str) -> str:
        if self.world_size <= 1:
            return value
        payload = [value]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _broadcast_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if self.world_size <= 1:
            return payload or {}
        data = [payload]
        dist.broadcast_object_list(data, src=0)
        result = data[0]
        if result is None:
            raise RuntimeError("Failed to broadcast job payload")
        return result

    def _setup_distributed(self) -> tuple[int, int]:
        if dist.is_available():
            if not dist.is_initialized() and "RANK" in os.environ:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
            if dist.is_initialized():
                if torch.cuda.is_available():
                    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
                return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _barrier(self) -> None:
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _generate_job_id() -> str:
        return datetime.utcnow().strftime("job-%Y%m%d%H%M%S%f")


__all__ = ["AnimateService"]
