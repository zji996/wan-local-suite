from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AnimateSettings(BaseSettings):
    checkpoint_dir: Optional[Path] = None
    models_root: Path = Path("models")
    model_id: str = "Wan-AI/Wan2.2-Animate-14B"
    model_revision: Optional[str] = None
    jobs_root: Path = Path("data/jobs")
    default_sample_solver: str = "dpm++"
    default_sampling_steps: int = 20
    default_clip_len: int = 77
    default_refert_num: int = 1
    default_shift: float = 5.0
    default_guide_scale: float = 1.0
    default_seed: int = -1
    default_offload_model: bool = True
    default_fps: int = 30
    enable_sequence_parallel: Optional[bool] = None
    enable_fsdp: bool = False
    use_relighting_lora: bool = False
    init_on_cpu: bool = True
    device_id: int = 0
    config_module: str = "third_party.wan2.2.wan.configs.wan_animate_14B"
    config_attr: str = "animate_14B"

    model_config = SettingsConfigDict(
        env_prefix="WAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> AnimateSettings:
    return AnimateSettings()
