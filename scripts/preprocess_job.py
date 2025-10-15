#!/usr/bin/env python3
"""Convert a YAML job spec into Wan-Animate preprocessing assets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic import ValidationInfo

from backend.app.config import AnimateSettings, get_settings


class Resolution(BaseModel):
    width: int = Field(1280, description="Target width for preprocessing")
    height: int = Field(720, description="Target height for preprocessing")

    @field_validator("width", "height")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("resolution dimensions must be positive")
        return value


class MaskSettings(BaseModel):
    iterations: int = 3
    k: int = 7
    w_len: int = 1
    h_len: int = 1

    @field_validator("iterations", "k", "w_len", "h_len")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("mask parameters must be non-negative")
        return value


class JobSpec(BaseModel):
    job_id: str = Field(..., description="Identifier used for the output folder")
    mode: Literal["animate", "replace"] = "animate"
    video: Path = Field(..., description="Path to the driving video")
    reference_image: Path = Field(..., description="Path to the reference image")
    process_checkpoint: Optional[Path] = Field(
        None,
        description="Directory containing preprocess checkpoints (pose2d/, det/, sam2/, ...)",
    )
    output_root: Path = Field(Path("data/jobs"), description="Base directory for job artifacts")
    resolution: Resolution = Field(default_factory=Resolution)
    fps: int = 30
    retarget: bool = False
    use_flux: bool = False
    mask: MaskSettings = Field(default_factory=MaskSettings)

    @field_validator("fps")
    @classmethod
    def _validate_fps(cls, value: int) -> int:
        if value == -1 or value > 0:
            return value
        raise ValueError("fps must be -1 or a positive integer")

    @field_validator("video", "reference_image", mode="before")
    @classmethod
    def _expand_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("output_root", "process_checkpoint", mode="before")
    @classmethod
    def _expand_optional_path(cls, value: Optional[Path | str]) -> Optional[Path]:
        if value is None:
            return None
        return Path(value).expanduser().resolve()

    @field_validator("use_flux")
    @classmethod
    def _flux_requires_retarget(cls, value: bool, info: ValidationInfo) -> bool:
        retarget = info.data.get("retarget", False)
        if value and not retarget:
            raise ValueError("use_flux=true requires retarget=true")
        return value


ROOT_DIR = Path(__file__).resolve().parent.parent
PREPROCESS_SCRIPT = (
    ROOT_DIR
    / "third_party"
    / "wan2.2"
    / "wan"
    / "modules"
    / "animate"
    / "preprocess"
    / "preprocess_data.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML job spec")
    parser.add_argument(
        "--dump-resolved",
        action="store_true",
        help="Print the resolved configuration as JSON for debugging",
    )
    return parser.parse_args()


def load_job_spec(path: Path) -> JobSpec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # noqa: F841
        raise FileNotFoundError(f"Job config not found: {path}") from exc
    except yaml.YAMLError as exc:  # type: ignore[assignment]
        raise ValueError(f"Failed to parse YAML: {exc}") from exc

    if data is None:
        raise ValueError("Job config is empty")

    try:
        return JobSpec.model_validate(data)
    except ValidationError as exc:  # noqa: F841
        raise ValueError(f"Invalid job config: {exc}") from exc


def resolve_checkpoint(spec: JobSpec, settings: AnimateSettings) -> Path:
    candidates: list[Path] = []

    if spec.process_checkpoint is not None:
        candidates.append(spec.process_checkpoint)

    if settings.checkpoint_dir is not None:
        candidates.append(settings.checkpoint_dir)
        candidates.append(settings.checkpoint_dir / "process_checkpoint")

    model_suffix = settings.model_id.split("/")[-1]
    models_root = settings.models_root.expanduser().resolve()
    candidates.append(models_root / model_suffix / "process_checkpoint")
    candidates.append(models_root / model_suffix)

    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate.expanduser().resolve()
        if not path.exists():
            continue
        if (path / "pose2d").exists() and (path / "det").exists():
            return path
        # Sometimes the path already points at Wan2.2-Animate-14B, but the
        # checkpoints live under the process_checkpoint subdirectory.
        nested = path / "process_checkpoint"
        if nested.exists() and (nested / "pose2d").exists() and (nested / "det").exists():
            return nested

    raise FileNotFoundError(
        "Unable to locate preprocess checkpoints. Provide process_checkpoint in the job spec "
        "or set WAN_CHECKPOINT_DIR to a Wan2.2-Animate-14B download."
    )


def run_preprocess(spec: JobSpec, checkpoint_dir: Path) -> None:
    output_dir = (spec.output_root / spec.job_id).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store the resolved configuration next to the assets for reproducibility.
    (output_dir / "preprocess_config.json").write_text(
        json.dumps(spec_to_dataclass(spec), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--ckpt_path",
        str(checkpoint_dir),
        "--video_path",
        str(spec.video),
        "--refer_path",
        str(spec.reference_image),
        "--save_path",
        str(output_dir),
        "--resolution_area",
        str(spec.resolution.width),
        str(spec.resolution.height),
        "--fps",
        str(spec.fps),
    ]

    if spec.mode == "replace":
        cmd.append("--replace_flag")
        cmd.extend(["--iterations", str(spec.mask.iterations)])
        cmd.extend(["--k", str(spec.mask.k)])
        cmd.extend(["--w_len", str(spec.mask.w_len)])
        cmd.extend(["--h_len", str(spec.mask.h_len)])

    if spec.retarget:
        cmd.append("--retarget_flag")
    if spec.use_flux:
        cmd.append("--use_flux")

    env = os.environ.copy()

    subprocess.run(cmd, check=True, cwd=ROOT_DIR, env=env)


def spec_to_dataclass(spec: JobSpec):
    """Convert a Pydantic model to a nested dataclass-like structure for JSON dumping."""
    def _convert(value):
        if isinstance(value, BaseModel):
            return {k: _convert(v) for k, v in value.model_dump(mode="python").items()}
        if isinstance(value, Path):
            return str(value)
        return value

    return {k: _convert(v) for k, v in spec.model_dump(mode="python").items()}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    spec = load_job_spec(config_path)

    if not spec.video.exists():
        raise FileNotFoundError(f"Driving video not found: {spec.video}")
    if not spec.reference_image.exists():
        raise FileNotFoundError(f"Reference image not found: {spec.reference_image}")

    settings = get_settings()
    checkpoint_dir = resolve_checkpoint(spec, settings)

    if args.dump_resolved:
        print(json.dumps(spec_to_dataclass(spec), indent=2, ensure_ascii=False))
        print(f"Using preprocess checkpoints at: {checkpoint_dir}")

    run_preprocess(spec, checkpoint_dir)

    output_dir = (spec.output_root / spec.job_id).expanduser().resolve()
    print(f"Preprocess assets written to {output_dir}")


if __name__ == "__main__":
    main()
