"""Client adapter for MiniMax H3 Turbo Backend Service.

Connects to the standalone backend service over HTTP REST API.
Exposes the exact same interface (constants, functions, data structures)
as backend.py so that frontend.py can run in fast, zero-reload mode.
"""
import copy
import glob
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# Default backend URL (configurable via environment variable)
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _fmt_secs(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def is_service_alive(url: str = DEFAULT_BACKEND_URL, timeout: float = 0.8) -> bool:
    """Check if the backend service is reachable and healthy."""
    try:
        req = urllib.request.Request(f"{url}/health", headers={"User-Agent": "AIQ-Frontend-Client"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "ready"
    except Exception:
        pass
    return False


class _ProxyDict(dict):
    """Dictionary that refreshes itself from a callback before read operations."""
    def __init__(self, fetch_fn, initial_data=None):
        super().__init__(initial_data or {})
        self._fetch_fn = fetch_fn

    def _sync(self):
        try:
            fresh = self._fetch_fn()
            if isinstance(fresh, dict):
                super().clear()
                super().update(fresh)
        except Exception:
            pass

    def __getitem__(self, key):
        self._sync()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._sync()
        return super().get(key, default)

    def __contains__(self, key):
        self._sync()
        return super().__contains__(key)

    def copy(self):
        self._sync()
        return super().copy()

    def items(self):
        self._sync()
        return super().items()

    def values(self):
        self._sync()
        return super().values()

    def keys(self):
        self._sync()
        return super().keys()


class BackendClient:
    """Drop-in replacement for the in-process backend module."""

    def __init__(self, backend_url: str = DEFAULT_BACKEND_URL):
        self.backend_url = backend_url.rstrip("/")

        # Fetch engine configuration from the backend service
        config = self._http_get("/config")
        if not config:
            raise ConnectionError(f"Failed to connect to MiniMax backend service at {self.backend_url}")

        self.OUTPUT_DIR = config.get("OUTPUT_DIR", "/kaggle/working/outputs")
        self.ZIP_PROJECT_DIR = config.get("ZIP_PROJECT_DIR", "/kaggle/working/manifest_projects")
        self.SERVER_PORT = config.get("SERVER_PORT", 7860)
        self.LANGUAGES = config.get("LANGUAGES", ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian"])
        self.RESOLUTIONS = config.get("RESOLUTIONS", ["720p", "540p", "480p", "360p"])
        self.ASPECTS = config.get("ASPECTS", ["Auto (from image, else 16:9)", "16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"])
        self.DURATIONS = config.get("DURATIONS", {"5s (129 frames)": 129, "10s (257 frames)": 257, "15s (385 frames)": 385, "20s (513 frames)": 513})
        self.STEP_PRESETS = config.get("STEP_PRESETS", {"4 steps (Turbo default)": 4, "6 steps": 6, "8 steps": 8})
        self.DEFAULT_STEPS = config.get("DEFAULT_STEPS", "4 steps (Turbo default)")
        self.DEFAULT_SAMPLER = config.get("DEFAULT_SAMPLER", "euler")
        self.LORA_CATALOG = config.get("LORA_CATALOG", {})
        self.LORA_CHOICES = config.get("LORA_CHOICES", ["(None)"])
        self.NO_LORA = config.get("NO_LORA", "(None)")
        self.REFERENCES_SUPPORTED = config.get("REFERENCES_SUPPORTED", False)
        self.VARIANT_LABEL = config.get("VARIANT_LABEL", "MiniMax H3 Turbo")
        self.INT8_LABEL = config.get("INT8_LABEL", {"kitchen": "Comfy Kitchen INT8 tensor cores"})

        # Shared upload directory for serializing images across processes on same machine
        self.UPLOAD_DIR = os.path.join(self.OUTPUT_DIR, ".temp_uploads")
        try:
            os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        except Exception:
            pass

        # Dynamic proxies for JOB and PROGRESS dictionaries
        self.JOB = _ProxyDict(self._fetch_job_state, {
            "id": 0, "done": 0, "running": False, "video": None, "status": "Ready.", "label": ""
        })
        self.PROGRESS = _ProxyDict(self._fetch_progress, {
            "active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0,
            "video": None, "message": ""
        })

        self.JOB_QUEUE = []
        self.JOB_HISTORY = []

    def _http_get(self, path: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        url = f"{self.backend_url}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIQ-Frontend-Client"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return None

    def _http_post(self, path: str, payload: Dict[str, Any] = None, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        url = f"{self.backend_url}{path}"
        try:
            data = json.dumps(payload or {}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "AIQ-Frontend-Client"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return None

    def _fetch_progress(self) -> Dict[str, Any]:
        data = self._http_get("/progress")
        return data or {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": ""}

    def _fetch_job_state(self) -> Dict[str, Any]:
        data = self._http_get("/queue")
        if data:
            self.JOB_QUEUE = data.get("queue", [])
            self.JOB_HISTORY = data.get("history", [])
            return {
                "id": data.get("current_id", 0),
                "done": 0,
                "running": data.get("running", False),
                "label": data.get("current_label", ""),
                "video": None,
                "status": "Running..." if data.get("running") else "Ready."
            }
        return {"id": 0, "done": 0, "running": False, "video": None, "status": "Ready.", "label": ""}

    def _serialize_arg(self, arg: Any) -> Any:
        """Convert PIL Images into temporary file paths accessible to the backend."""
        if isinstance(arg, Image.Image):
            os.makedirs(self.UPLOAD_DIR, exist_ok=True)
            path = os.path.join(self.UPLOAD_DIR, f"upload_{int(time.time()*1000)}_{os.getpid()}_{id(arg)}.png")
            arg.convert("RGB").save(path, format="PNG")
            return path
        elif isinstance(arg, list) and arg and isinstance(arg[0], Image.Image):
            paths = []
            os.makedirs(self.UPLOAD_DIR, exist_ok=True)
            for idx, img in enumerate(arg):
                path = os.path.join(self.UPLOAD_DIR, f"upload_{int(time.time()*1000)}_{os.getpid()}_{id(img)}_{idx}.png")
                img.convert("RGB").save(path, format="PNG")
                paths.append(path)
            return paths
        elif hasattr(arg, "name") and isinstance(arg.name, str):
            # Gradio File or tempfile wrapper
            return arg.name
        return arg

    def queue_job(self, args: Tuple[Any, ...], label: str = "") -> Tuple[int, int]:
        """Submit a generation job to the backend service.
        Returns (job_id, queue_position).
        """
        serialized = [self._serialize_arg(a) for a in args]
        payload = {"args": serialized, "label": label}
        res = self._http_post("/queue", payload=payload)
        if res and "job_id" in res:
            return res["job_id"], res.get("position", 0)
        return -1, -1

    def start_job(self, *args, label: str = "") -> Tuple[int, int]:
        """Backward-compatible entry point: queues or starts the generation."""
        return self.queue_job(args, label=label)

    def clear_queue(self) -> int:
        """Clear all pending jobs from the queue without interrupting the running job."""
        res = self._http_post("/clear_queue")
        return res.get("cleared", 0) if res else 0

    def get_queue_status(self) -> Dict[str, Any]:
        """Fetch real-time queue status from the backend service."""
        res = self._http_get("/queue")
        if res:
            self.JOB_QUEUE = res.get("queue", [])
            self.JOB_HISTORY = res.get("history", [])
            return res
        return {
            "running": False,
            "current_id": 0,
            "current_label": "",
            "queue_len": 0,
            "queue": [],
            "history": []
        }

    def stop_generation(self) -> str:
        """Interrupt active generation and clear pending queue."""
        res = self._http_post("/stop")
        return res.get("message", "⏹️ Stop request sent.") if res else "⚠️ Failed to reach backend service."

    def load_latest_video(self) -> Tuple[Optional[str], str]:
        """Find the newest generated video from the backend service or local disk."""
        res = self._http_get("/latest_video")
        if res and res.get("video"):
            return res["video"], res.get("status", f"📁 Loaded latest video: {res['video']}")
        # Fallback to local disk scan
        videos = sorted(glob.glob(os.path.join(self.OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
        if not videos:
            return None, f"No videos in {self.OUTPUT_DIR} yet."
        return videos[-1], f"📁 Loaded latest video: {videos[-1]}"
