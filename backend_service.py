"""MiniMax H3 Turbo Standalone Backend Service - AIQUEST Academy.

Runs the heavy video model engine as a persistent backend daemon on localhost:8000.
Keeps 20B Transformer, Qwen text encoder, and VAEs resident in memory.
Allows frontend.py to be started, stopped, edited, and restarted instantly (in ~1s)
without reloading the multi-gigabyte models into GPU VRAM.
"""
import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional
from PIL import Image

import uvicorn
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        USE_MOCK = os.environ.get("MOCK", "").strip().lower() in ("true", "1", "yes")
        print("=" * 70)
        if USE_MOCK:
            print("🧪 Starting Backend Service in MOCK mode...")
            import backend_mock as eng
        else:
            print("🚀 Starting Backend Service with real MiniMax H3 engine...")
            import backend as eng
        print(f"🎬 Engine ready! Variant: {eng.VARIANT_LABEL}")
        print("=" * 70)
        _engine = eng
    return _engine


def start_in_background(log_file="/tmp/backend_service.log", host="127.0.0.1", port=8000):
    """Convenience helper to safely launch the backend service in the background from Jupyter/Python."""
    from start_service import main as start_main
    os.environ["BACKEND_HOST"] = str(host)
    os.environ["BACKEND_PORT"] = str(port)
    os.environ["BACKEND_LOG"] = str(log_file)
    start_main()

app = FastAPI(
    title="MiniMax H3 Turbo Backend Service",
    description="Persistent model inference & job queue server for Kaggle / GPU environments",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _deserialize_arg(arg: Any) -> Any:
    """If the argument is a path to an image, load it into a PIL Image."""
    if isinstance(arg, str) and os.path.isfile(arg):
        ext = os.path.splitext(arg)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            try:
                with Image.open(arg) as img:
                    return img.convert("RGB").copy()
            except Exception:
                return arg
    elif isinstance(arg, list) and arg and isinstance(arg[0], str) and os.path.isfile(arg[0]):
        ext = os.path.splitext(arg[0])[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            images = []
            for item in arg:
                if isinstance(item, str) and os.path.isfile(item):
                    try:
                        with Image.open(item) as img:
                            images.append(img.convert("RGB").copy())
                    except Exception:
                        images.append(item)
                else:
                    images.append(item)
            return images
    return arg


@app.get("/health")
@app.get("/status")
def health_endpoint():
    engine = get_engine()
    status_info = engine.get_queue_status()
    return {
        "status": "ready",
        "running": status_info["running"],
        "current_id": status_info["current_id"],
        "current_label": status_info["current_label"],
        "queue_len": status_info["queue_len"],
        "model_variant": engine.VARIANT_LABEL,
        "references_supported": engine.REFERENCES_SUPPORTED,
        "output_dir": engine.OUTPUT_DIR,
        "zip_project_dir": engine.ZIP_PROJECT_DIR,
        "server_port": engine.SERVER_PORT,
    }


@app.get("/config")
def config_endpoint():
    """Return all engine constants needed by the frontend UI."""
    engine = get_engine()
    return {
        "OUTPUT_DIR": engine.OUTPUT_DIR,
        "ZIP_PROJECT_DIR": engine.ZIP_PROJECT_DIR,
        "SERVER_PORT": engine.SERVER_PORT,
        "LANGUAGES": engine.LANGUAGES,
        "RESOLUTIONS": engine.RESOLUTIONS,
        "ASPECTS": engine.ASPECTS,
        "DURATIONS": engine.DURATIONS,
        "STEP_PRESETS": engine.STEP_PRESETS,
        "DEFAULT_STEPS": engine.DEFAULT_STEPS,
        "DEFAULT_SAMPLER": engine.DEFAULT_SAMPLER,
        "LORA_CATALOG": engine.LORA_CATALOG,
        "LORA_CHOICES": engine.LORA_CHOICES,
        "NO_LORA": engine.NO_LORA,
        "REFERENCES_SUPPORTED": engine.REFERENCES_SUPPORTED,
        "VARIANT_LABEL": engine.VARIANT_LABEL,
        "INT8_LABEL": engine.INT8_LABEL,
    }


@app.get("/queue")
def queue_endpoint():
    return get_engine().get_queue_status()


@app.post("/queue")
def queue_job_endpoint(payload: Dict[str, Any] = Body(...)):
    engine = get_engine()
    raw_args = payload.get("args", [])
    label = payload.get("label", "")
    deserialized_args = tuple(_deserialize_arg(a) for a in raw_args)
    job_id, pos = engine.queue_job(deserialized_args, label=label)
    return {"job_id": job_id, "position": pos}


@app.post("/stop")
def stop_endpoint():
    msg = get_engine().stop_generation()
    return {"message": msg}


@app.post("/clear_queue")
def clear_queue_endpoint():
    cleared = get_engine().clear_queue()
    return {"cleared": cleared}


@app.get("/progress")
def progress_endpoint():
    engine = get_engine()
    end = engine.PROGRESS["finished"] or time.time()
    elapsed = end - engine.PROGRESS["started"] if engine.PROGRESS["started"] else 0.0
    return {
        **engine.PROGRESS,
        "elapsed": elapsed,
        "queue_len": len(engine.JOB_QUEUE)
    }


@app.get("/latest_video")
def latest_video_endpoint():
    video, status = get_engine().load_latest_video()
    return {"video": video, "status": status}


@app.get("/videos")
def list_videos_endpoint():
    """List all generated videos with file metadata."""
    engine = get_engine()
    search_dirs = [engine.OUTPUT_DIR]
    if os.path.isdir("/kaggle/working/outputs") and os.path.realpath("/kaggle/working/outputs") != os.path.realpath(engine.OUTPUT_DIR):
        search_dirs.append("/kaggle/working/outputs")

    seen = set()
    videos = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.mp4")):
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                try:
                    size_mb = os.path.getsize(p) / (1024 * 1024)
                    mtime = os.path.getmtime(p)
                    date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                    videos.append({
                        "name": os.path.basename(p),
                        "path": p,
                        "size_mb": round(size_mb, 2),
                        "mtime": mtime,
                        "date": date_str
                    })
                except Exception:
                    videos.append({
                        "name": os.path.basename(p),
                        "path": p,
                        "size_mb": 0.0,
                        "mtime": 0,
                        "date": ""
                    })
    videos.sort(key=lambda x: x["mtime"], reverse=True)
    return {"videos": videos}


@app.get("/download/{filename}")
def download_endpoint(filename: str, download: int = 0):
    engine = get_engine()
    safe_name = os.path.basename(filename)
    search_dirs = [engine.OUTPUT_DIR]
    if os.path.isdir("/kaggle/working/outputs"):
        search_dirs.append("/kaggle/working/outputs")

    target = None
    for d in search_dirs:
        candidate = os.path.join(d, safe_name)
        if os.path.isfile(candidate):
            target = candidate
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found")

    disposition = "attachment" if download else "inline"
    return FileResponse(
        target,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'{disposition}; filename="{safe_name}"',
            "Cache-Control": "no-store"
        }
    )


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Backend Service")
    parser.add_argument("--host", default=os.environ.get("BACKEND_HOST", "127.0.0.1"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BACKEND_PORT", 8000)), help="Port to listen on")
    args = parser.parse_args()

    engine = get_engine()

    print("\n" + "=" * 70)
    print("🚀 MiniMax H3 Turbo Backend Service is ONLINE!")
    print("=" * 70)
    print(f"📡 API Address:      http://{args.host}:{args.port}")
    print(f"📁 Video Outputs:    {engine.OUTPUT_DIR}")
    print(f"💡 Frontend Launch:  Run 'python frontend.py' in another terminal/cell.")
    print("                      You can edit and rerun frontend.py anytime in ~1s without reloading models.")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
