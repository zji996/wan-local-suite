from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AnimateJobCreate(BaseModel):
    src_root_path: Path = Field(..., description="Directory containing prepared source assets.")
    job_id: Optional[str] = Field(None, description="Optional identifier to reuse an existing job directory.")
    replace_flag: bool = False
    clip_len: Optional[int] = None
    refert_num: Optional[int] = None
    shift: Optional[float] = None
    sample_solver: Optional[str] = None
    sampling_steps: Optional[int] = None
    guide_scale: Optional[float] = None
    input_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    offload_model: Optional[bool] = None
    fps: Optional[int] = None

    @field_validator("src_root_path")
    @classmethod
    def _expand_src_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class AnimateJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    output_path: Optional[Path] = None
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
