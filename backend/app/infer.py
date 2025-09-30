from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

from .config import get_settings
from .schemas.animate import AnimateJobCreate
from .services.animate_service import AnimateService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wan Animate inference job")
    parser.add_argument("--src-root", required=True, help="Directory containing src_pose.mp4/src_face.mp4/src_ref.png")
    parser.add_argument("--job-id", help="Optional job identifier")
    parser.add_argument("--replace", action="store_true", help="Enable character replacement pipeline")
    parser.add_argument("--clip-len", type=int, help="Clip length (must be 4n+1)" )
    parser.add_argument("--refert-num", type=int, help="Temporal guidance frame count (1 or 5)")
    parser.add_argument("--shift", type=float, help="Noise schedule shift parameter")
    parser.add_argument("--solver", choices=["dpm++", "unipc"], help="Sampling solver")
    parser.add_argument("--sampling-steps", type=int, help="Diffusion sampling steps")
    parser.add_argument("--guide-scale", type=float, help="Classifier-free guidance scale")
    parser.add_argument("--input-prompt", help="Text prompt override")
    parser.add_argument("--negative-prompt", help="Negative prompt override")
    parser.add_argument("--seed", type=int, help="Random seed (-1 for random)")
    parser.add_argument("--no-offload", action="store_true", help="Disable model offloading to CPU between stages")
    parser.add_argument("--fps", type=int, help="Output video FPS")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    settings = get_settings()
    service = AnimateService(settings=settings)

    offload_model: Optional[bool]
    offload_model = False if args.no_offload else None

    request = AnimateJobCreate(
        src_root_path=args.src_root,
        job_id=args.job_id,
        replace_flag=args.replace,
        clip_len=args.clip_len,
        refert_num=args.refert_num,
        shift=args.shift,
        sample_solver=args.solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        input_prompt=args.input_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=offload_model,
        fps=args.fps,
    )

    status = service.run_job_sync(request)
    if service.rank == 0:
        print(json.dumps(status.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
