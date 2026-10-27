"""
Backend Service for LTX-2.3 Audio-to-Video Engine.
Provides a RESTful Video Generation API and Queuing Service.
"""

import os
import sys
import time
import argparse
import logging
import tempfile
import threading
import subprocess
import re
import stat
import urllib.request
from typing import Optional, List, Dict, Any

# Ensure local imports work regardless of current working directory
ORIG_DIR = os.path.dirname(os.path.abspath(__file__))
if ORIG_DIR not in sys.path:
    sys.path.insert(0, ORIG_DIR)

from queue_service import (
    queue_service,
    STATUS_WAITING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    OUTPUT_DIR,
)
from video_engine import video_engine, NUM_GPUS, _GPU_SM

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
logger = logging.getLogger("backend_service")

# Try importing FastAPI
try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Run 'pip install fastapi uvicorn python-multipart' to enable HTTP API server.")


# ---- Pydantic Models (when FastAPI available) ----
if HAS_FASTAPI:
    class GenerateRequest(BaseModel):
        prompt: str = Field(..., description="Visual and dialogue prompt")
        image_path: Optional[str] = Field(None, description="Local path to start/portrait image")
        audio_path: Optional[str] = Field(None, description="Local path to audio track")
        seed: int = Field(-1, description="Random seed (-1 for random)")
        duration: str = Field("5 Seconds (121 frames)", description="Duration label")
        resolution: str = Field("720p", description="Resolution: 1080p, 720p, 540p, or 480p")
        aspect_ratio: Optional[str] = Field(None, description="Aspect ratio label")
        aspectRatio: Optional[str] = Field(None, description="Aspect ratio label (camelCase)")
        match_audio_dur: Optional[bool] = Field(None, description="Snap duration to audio track length")
        matchAudioDuration: Optional[bool] = Field(None, description="Snap duration to audio track length (camelCase)")

    class JobResponse(BaseModel):
        id: str
        prompt: str
        status: str
        progress: float
        progress_desc: str
        result_video_path: Optional[str] = None
        video_url: Optional[str] = None
        error: Optional[str] = None
        created_at: float
        started_at: Optional[float] = None
        completed_at: Optional[float] = None
        execution_time: Optional[float] = None


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn python-multipart")

    app = FastAPI(
        title="LTX-2.3 Audio-to-Video API",
        description="High-performance backend service with video generation API and queuing service for LTX-2.3 22B Distilled.",
        version="2.3.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def startup_event():
        logger.info("Starting Queue Service worker...")
        queue_service.start_worker(video_engine)

    @app.on_event("shutdown")
    def shutdown_event():
        logger.info("Stopping Queue Service worker...")
        queue_service.stop_worker()

    @app.get("/")
    def index():
        return {
            "service": "LTX-2.3 Audio-to-Video Backend",
            "status": "online",
            "api_docs": "/docs",
            "gpus": NUM_GPUS,
            "engine_loaded": video_engine.is_loaded,
        }

    @app.get("/api/health")
    def health_check():
        import psutil
        ram = psutil.virtual_memory()
        gpu_info = []
        try:
            import torch
            for i in range(NUM_GPUS):
                free_gb = torch.cuda.mem_get_info(i)[0] / 1024**3
                total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_info.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "free_vram_gb": round(free_gb, 2),
                    "total_vram_gb": round(total_gb, 2),
                })
        except Exception:
            pass

        waiting_jobs = queue_service.list_jobs(status_filter=STATUS_WAITING)
        running_jobs = queue_service.list_jobs(status_filter=STATUS_RUNNING)

        return {
            "status": "healthy",
            "model_loaded": video_engine.is_loaded,
            "ram_available_gb": round(ram.available / 1024**3, 1),
            "ram_total_gb": round(ram.total / 1024**3, 1),
            "gpus": gpu_info,
            "queue": {
                "waiting_count": len(waiting_jobs),
                "running_count": len(running_jobs),
                "is_busy": len(running_jobs) > 0,
            },
        }

    # ---- Video Generation Endpoints ----
    @app.post("/api/generate", response_model=Dict[str, Any])
    async def submit_generate_json(req: GenerateRequest):
        """Submit a new video generation job via JSON."""
        resolved_aspect = req.aspect_ratio or req.aspectRatio or "Auto (match reference image)"
        resolved_match = req.match_audio_dur if req.match_audio_dur is not None else (
            req.matchAudioDuration if req.matchAudioDuration is not None else False
        )
        job = queue_service.register_job(
            prompt=req.prompt,
            image_path=req.image_path,
            audio_path=req.audio_path,
            seed=req.seed,
            duration=req.duration,
            resolution=req.resolution,
            aspect_ratio=resolved_aspect,
            match_audio_dur=resolved_match,
        )
        job_data = job.to_dict()
        if job.result_video_path:
            job_data["video_url"] = f"/api/videos/{os.path.basename(job.result_video_path)}"
        return {"status": "queued", "job_id": job.id, "job": job_data}

    @app.post("/api/generate/upload")
    async def submit_generate_upload(
        prompt: str = Form(...),
        seed: int = Form(-1),
        duration: str = Form("5 Seconds (121 frames)"),
        resolution: str = Form("720p"),
        aspect_ratio: Optional[str] = Form(None),
        aspectRatio: Optional[str] = Form(None),
        match_audio_dur: Optional[bool] = Form(None),
        matchAudioDuration: Optional[bool] = Form(None),
        image_file: Optional[UploadFile] = File(None),
        audio_file: Optional[UploadFile] = File(None),
    ):
        """Submit a video generation job with uploaded image/audio files."""
        resolved_aspect = aspect_ratio or aspectRatio or "Auto (match reference image)"
        resolved_match = match_audio_dur if match_audio_dur is not None else (
            matchAudioDuration if matchAudioDuration is not None else False
        )
        saved_image_path = None
        saved_audio_path = None

        upload_dir = os.path.join(OUTPUT_DIR, ".uploads")
        os.makedirs(upload_dir, exist_ok=True)

        if image_file and image_file.filename:
            suffix = os.path.splitext(image_file.filename)[1] or ".png"
            tmp_img = os.path.join(upload_dir, f"img_{int(time.time())}_{image_file.filename}")
            content = await image_file.read()
            with open(tmp_img, "wb") as f:
                f.write(content)
            saved_image_path = tmp_img

        if audio_file and audio_file.filename:
            suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
            tmp_aud = os.path.join(upload_dir, f"aud_{int(time.time())}_{audio_file.filename}")
            content = await audio_file.read()
            with open(tmp_aud, "wb") as f:
                f.write(content)
            saved_audio_path = tmp_aud

        job = queue_service.register_job(
            prompt=prompt,
            image_path=saved_image_path,
            audio_path=saved_audio_path,
            seed=seed,
            duration=duration,
            resolution=resolution,
            aspect_ratio=resolved_aspect,
            match_audio_dur=resolved_match,
        )
        return {"status": "queued", "job_id": job.id, "job": job.to_dict()}

    @app.post("/api/generate/sync")
    async def generate_sync(req: GenerateRequest, timeout_seconds: int = Query(600, ge=30, le=1800)):
        """
        Synchronously wait for a generation job to complete.
        Useful for simple HTTP clients that want to block until ready.
        """
        job = queue_service.register_job(
            prompt=req.prompt,
            image_path=req.image_path,
            audio_path=req.audio_path,
            seed=req.seed,
            duration=req.duration,
            resolution=req.resolution,
            aspect_ratio=req.aspect_ratio,
            match_audio_dur=req.match_audio_dur,
        )
        t_start = time.time()
        while time.time() - t_start < timeout_seconds:
            current_job = queue_service.get_job(job.id)
            if not current_job:
                raise HTTPException(status_code=404, detail="Job disappeared")
            if current_job.status == STATUS_COMPLETED:
                res = current_job.to_dict()
                if current_job.result_video_path:
                    res["video_url"] = f"/api/videos/{os.path.basename(current_job.result_video_path)}"
                return res
            elif current_job.status in (STATUS_FAILED, STATUS_CANCELLED):
                raise HTTPException(status_code=500, detail=f"Job {current_job.status}: {current_job.error}")
            time.sleep(1.5)

        raise HTTPException(status_code=504, detail="Generation timed out")

    # ---- Job & Queue Management Endpoints ----
    @app.get("/api/jobs")
    def list_jobs(status: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=200)):
        """List recent jobs."""
        jobs = queue_service.list_jobs(status_filter=status, limit=limit)
        results = []
        for j in jobs:
            jd = j.to_dict()
            if j.result_video_path:
                jd["video_url"] = f"/api/videos/{os.path.basename(j.result_video_path)}"
            results.append(jd)
        return results

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        """Get status and details of a specific job."""
        job = queue_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        jd = job.to_dict()
        if job.result_video_path:
            jd["video_url"] = f"/api/videos/{os.path.basename(job.result_video_path)}"
        return jd

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        """Cancel a waiting or running job."""
        success = queue_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not active or not found")
        return {"status": "cancelled", "job_id": job_id}

    @app.post("/api/queue/clear")
    def clear_queue():
        """Cancel all waiting jobs in queue."""
        count = queue_service.clear_waiting_jobs()
        return {"status": "cleared", "cancelled_jobs_count": count}

    @app.get("/api/queue/status")
    def get_queue_status():
        """Get formatted text representation and counts of current queue."""
        waiting = queue_service.list_jobs(status_filter=STATUS_WAITING)
        running = queue_service.list_jobs(status_filter=STATUS_RUNNING)
        return {
            "status_text": queue_service.get_queue_status_text(),
            "waiting_count": len(waiting),
            "running_count": len(running),
            "is_busy": len(running) > 0,
        }

    # ---- Video Assets Endpoints ----
    @app.get("/api/videos")
    def list_videos(limit: int = Query(20, ge=1, le=100)):
        """List finished video files."""
        vids = queue_service.get_finished_videos(limit=limit)
        return [
            {
                "filename": os.path.basename(v),
                "path": v,
                "url": f"/api/videos/{os.path.basename(v)}",
                "size_mb": round(os.path.getsize(v) / 1024**2, 2) if os.path.exists(v) else 0,
                "created_at": os.path.getmtime(v) if os.path.exists(v) else 0,
            }
            for v in vids
        ]

    @app.get("/api/videos/{filename}")
    def get_video_file(filename: str):
        """Stream or download a finished video file."""
        safe_filename = os.path.basename(filename)

        # 1. Check all registered jobs for matching output filename
        for job in queue_service.list_jobs(limit=100):
            if job.result_video_path and os.path.basename(job.result_video_path) == safe_filename:
                if os.path.isfile(job.result_video_path):
                    return FileResponse(job.result_video_path, media_type="video/mp4", filename=safe_filename)

        # 2. Check all candidate output directories
        for d in queue_service.get_output_dirs():
            candidate = os.path.join(d, safe_filename)
            if os.path.isfile(candidate):
                return FileResponse(candidate, media_type="video/mp4", filename=safe_filename)

        raise HTTPException(status_code=404, detail=f"Video file {safe_filename} not found")

    return app


def start_cloudflare_tunnel(port: int):
    """Start Cloudflare Quick Tunnel for public access (optional fallback)."""
    import re
    import stat
    import urllib.request

    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            print("Downloading Cloudflare tunnel binary...")
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary
            )
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen(
            [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {e}")
        return

    def _watch():
        for line in tunnel.stdout:
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match:
                print(f"  * Backend Cloudflare tunnel URL: {match.group(0)}")
                sys.stdout.flush()
                break
        for _ in tunnel.stdout:
            pass

    threading.Thread(target=_watch, daemon=True).start()


def run_backend(host: str = "0.0.0.0", port: int = 8000, tunnel: bool = False):
    """Run the backend service server."""
    if not HAS_FASTAPI:
        print("❌ Cannot start backend service: FastAPI is not installed.")
        print("Run: pip install fastapi uvicorn python-multipart")
        return

    import uvicorn

    if tunnel:
        start_cloudflare_tunnel(port)

    print(f"\n🚀 Starting LTX Video Generation API Backend on http://{host}:{port} ...")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2.3 Video Generation Backend Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--tunnel", action="store_true", help="Launch Cloudflare quick tunnel")
    args = parser.parse_args()

    run_backend(host=args.host, port=args.port, tunnel=args.tunnel)
