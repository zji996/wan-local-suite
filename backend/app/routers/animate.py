from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from ..schemas.animate import AnimateJobCreate, AnimateJobStatus
from ..services.animate_service import AnimateService

router = APIRouter(prefix="/animate", tags=["animate"])
service = AnimateService()


@router.post('/jobs', response_model=AnimateJobStatus, status_code=202)
async def create_job(request: AnimateJobCreate, background_tasks: BackgroundTasks) -> AnimateJobStatus:
    try:
        status = service.create_job(request)
    except FileExistsError as exc:  # noqa: F841
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(service.execute_job, status.job_id, request)
    return status


@router.post('/run', response_model=AnimateJobStatus)
async def run_job(request: AnimateJobCreate) -> AnimateJobStatus:
    try:
        return service.run_job_sync(request)
    except FileNotFoundError as exc:  # noqa: F841
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/jobs/{job_id}', response_model=AnimateJobStatus)
async def get_job(job_id: str) -> AnimateJobStatus:
    try:
        return service.get_status(job_id)
    except FileNotFoundError as exc:  # noqa: F841
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/jobs/{job_id}/media')
async def download_job_media(job_id: str) -> FileResponse:
    status = service.get_status(job_id)
    if not status.output_path:
        raise HTTPException(status_code=404, detail="Output not available yet")
    return FileResponse(status.output_path, filename=f"{job_id}.mp4")
