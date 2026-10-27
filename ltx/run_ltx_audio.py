"""
Unified Launcher for LTX-2.3 Audio-to-Video Engine.
Coordinates the Backend Video Generation API, Queuing Service, and Gradio Frontend.

Maintains complete backward compatibility with existing notebooks and scripts:
- python run_ltx_audio.py               (Launches both backend API and Gradio UI)
- python run_ltx_audio.py --backend-only (Launches FastAPI backend service only)
- python run_ltx_audio.py --frontend-only(Launches Gradio frontend only)
"""

import os
import sys
import argparse
import threading
import time

# Ensure local imports work regardless of current working directory
ORIG_DIR = os.path.dirname(os.path.abspath(__file__))
if ORIG_DIR not in sys.path:
    sys.path.insert(0, ORIG_DIR)

from queue_service import (
    queue_service,
    Job,
    OUTPUT_DIR,
    STATUS_WAITING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
)
from video_engine import (
    video_engine,
    LTXVideoEngine,
    get_resolution,
    snap_to_ltx_frames,
    AUTO_ASPECT,
    NUM_GPUS,
    _GPU_SM,
)
from frontend import (
    create_ui,
    launch_frontend,
    start_cloudflare_tunnel,
    video_generation_fn as Video_Generation,
    register_job_fn as register_job,
    clear_waiting_jobs_fn as clear_waiting_jobs,
    get_queue_status_fn as get_queue_status,
    get_finished_videos_fn as get_finished_videos,
    load_latest_video_fn as load_latest_video,
)
from backend import HAS_FASTAPI, run_backend, create_app

# Backward compatibility wrappers for shared queue
queue_lock = queue_service._lock


def set_job_status(job_id, status):
    queue_service.set_job_status(job_id, status)


def remove_job(job_id):
    queue_service.remove_job(job_id)


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Audio-to-Video Launcher")
    parser.add_argument("--mode", choices=["both", "frontend", "backend"], default="both",
                        help="Run mode: 'both' (default), 'frontend', or 'backend'")
    parser.add_argument("--backend-only", action="store_true", help="Run only the backend API server")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the Gradio frontend")
    parser.add_argument("--api-port", type=int, default=8000, help="Backend API port (default 8000)")
    parser.add_argument("--ui-port", type=int, default=7860, help="Frontend Gradio port (default 7860)")
    parser.add_argument("--share", action="store_true", default=True, help="Enable Gradio share link")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable Cloudflare quick tunnel")
    args = parser.parse_args()

    mode = args.mode
    if args.backend_only:
        mode = "backend"
    elif args.frontend_only:
        mode = "frontend"

    use_tunnel = not args.no_tunnel

    print("=" * 65)
    print("🎬 LTX-2.3 Audio-to-Video Engine & Service")
    print(f"📡 Execution Mode: {mode.upper()}")
    print("=" * 65)

    if mode == "backend":
        run_backend(host="0.0.0.0", port=args.api_port, tunnel=use_tunnel)
        return

    if mode == "both":
        # Launch backend API in a background thread if FastAPI is available
        if HAS_FASTAPI:
            print(f"🚀 Starting background REST API on port {args.api_port}...")
            api_thread = threading.Thread(
                target=run_backend,
                kwargs={"host": "0.0.0.0", "port": args.api_port, "tunnel": False},
                daemon=True,
                name="BackendAPI",
            )
            api_thread.start()
            time.sleep(1.0)
        else:
            print("ℹ️ FastAPI not installed, running with integrated in-process queue service.")

    # Launch Gradio Frontend UI
    launch_frontend(server_port=args.ui_port, share=args.share, tunnel=use_tunnel)


if __name__ == "__main__":
    main()
