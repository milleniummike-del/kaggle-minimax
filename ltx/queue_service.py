"""
Queue Service for LTX-2.3 Audio-to-Video Generation.
Handles job registration, FIFO queuing, status tracking, progress reporting,
and cancellation in a thread-safe manner.
"""

import os
import glob
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

logger = logging.getLogger("queue_service")

# Default outputs folder
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Status constants
STATUS_WAITING = "waiting"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a video generation job in the queuing system."""
    id: str
    prompt: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    seed: int = -1
    duration: str = "5 Seconds (121 frames)"
    resolution: str = "720p"
    aspect_ratio: str = "Auto (match reference image)"
    match_audio_dur: bool = False

    # State
    status: str = STATUS_WAITING
    progress: float = 0.0
    progress_desc: str = ""
    result_video_path: Optional[str] = None
    error: Optional[str] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job details to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "audio_path": self.audio_path,
            "seed": self.seed,
            "duration": self.duration,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "match_audio_dur": self.match_audio_dur,
            "status": self.status,
            "progress": round(self.progress, 3),
            "progress_desc": self.progress_desc,
            "result_video_path": self.result_video_path,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": round(self.execution_time, 2) if self.execution_time is not None else None,
        }


class QueueService:
    """
    Thread-safe Queuing Service for sequential video generation jobs.
    Coordinates between API callers, frontend UI, and the video generation engine.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self._jobs: Dict[str, Job] = {}
        self._waiting_queue: List[str] = []  # list of job_ids
        self._current_job_id: Optional[str] = None
        self._cancel_flags: Dict[str, bool] = {}
        self._lock = threading.RLock()

        # Background worker state
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._job_available = threading.Event()
        self._engine = None

    def register_job(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        seed: int = -1,
        duration: str = "5 Seconds (121 frames)",
        resolution: str = "720p",
        aspect_ratio: str = "Auto (match reference image)",
        match_audio_dur: bool = False,
        job_id: Optional[str] = None,
    ) -> Job:
        """Register a new job and place it in the waiting queue."""
        with self._lock:
            jid = job_id or uuid.uuid4().hex[:8]
            clean_prompt = prompt or "(no prompt)"
            job = Job(
                id=jid,
                prompt=clean_prompt,
                image_path=image_path,
                audio_path=audio_path,
                seed=seed,
                duration=duration,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                match_audio_dur=match_audio_dur,
                status=STATUS_WAITING,
                progress=0.0,
                progress_desc="In queue",
            )
            self._jobs[jid] = job
            self._waiting_queue.append(jid)
            self._cancel_flags[jid] = False

            # Signal worker thread if active
            self._job_available.set()
            return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by its ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, status_filter: Optional[str] = None, limit: int = 50) -> List[Job]:
        """List jobs, optionally filtered by status, ordered newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
            if status_filter:
                jobs = [j for j in jobs if j.status == status_filter]
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def set_job_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Update status of a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            if error:
                job.error = error

            now = time.time()
            if status == STATUS_RUNNING:
                self._current_job_id = job_id
                job.started_at = now
                if job_id in self._waiting_queue:
                    self._waiting_queue.remove(job_id)
            elif status in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED):
                if self._current_job_id == job_id:
                    self._current_job_id = None
                if job_id in self._waiting_queue:
                    self._waiting_queue.remove(job_id)
                job.completed_at = now
                if job.started_at:
                    job.execution_time = now - job.started_at

    def update_progress(self, job_id: str, progress: float, desc: str = ""):
        """Update generation progress for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = max(0.0, min(1.0, float(progress)))
                if desc:
                    job.progress_desc = desc

    def set_result(self, job_id: str, video_path: str):
        """Set final video path and mark completed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.result_video_path = video_path
                self.set_job_status(job_id, STATUS_COMPLETED)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job. If waiting, immediately cancel. If running, flag for cancellation."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            self._cancel_flags[job_id] = True
            if job.status == STATUS_WAITING:
                if job_id in self._waiting_queue:
                    self._waiting_queue.remove(job_id)
                job.status = STATUS_CANCELLED
                job.progress_desc = "Cancelled by user"
                return True
            elif job.status == STATUS_RUNNING:
                job.progress_desc = "Stopping..."
                return True
            return False

    def clear_waiting_jobs(self) -> int:
        """Cancel all waiting jobs in queue."""
        with self._lock:
            cancelled_count = 0
            for jid in list(self._waiting_queue):
                job = self._jobs.get(jid)
                if job and job.status == STATUS_WAITING:
                    job.status = STATUS_CANCELLED
                    job.progress_desc = "Cancelled"
                    self._cancel_flags[jid] = True
                    cancelled_count += 1
            self._waiting_queue.clear()
            return cancelled_count

    def remove_job(self, job_id: str):
        """Remove a job from active queue view (backward compatibility)."""
        with self._lock:
            if job_id in self._waiting_queue:
                self._waiting_queue.remove(job_id)
            if self._current_job_id == job_id:
                self._current_job_id = None

    def is_cancelled(self, job_id: Optional[str]) -> bool:
        """Check if cancellation was requested for a job."""
        if not job_id:
            return False
        with self._lock:
            return self._cancel_flags.get(job_id, False)

    def get_queue_status_text(self) -> str:
        """
        Formatted text representation of current queue.
        Matches previous Gradio format:
        1. [running] prompt...
        2. [waiting] prompt...
        """
        with self._lock:
            active_entries = []
            # Running job first
            if self._current_job_id and self._current_job_id in self._jobs:
                running_job = self._jobs[self._current_job_id]
                active_entries.append(running_job)

            # Followed by waiting jobs
            for jid in self._waiting_queue:
                if jid in self._jobs:
                    job = self._jobs[jid]
                    if job not in active_entries:
                        active_entries.append(job)

            if not active_entries:
                return "Queue is empty"

            lines = []
            for i, j in enumerate(active_entries):
                pct = f" ({int(j.progress * 100)}%)" if j.status == STATUS_RUNNING and j.progress > 0 else ""
                lines.append(f"{i + 1}. [{j.status}{pct}] {j.prompt[:80]}")
            return "\n".join(lines)

    def get_finished_videos(self, limit: int = 12) -> List[str]:
        """Return paths of finished video files in outputs folder (newest first)."""
        pattern = os.path.join(self.output_dir, "*.mp4")
        vids = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        return vids[:limit]

    def load_latest_video(self) -> Tuple[Optional[str], str]:
        """Load path of most recently created video."""
        pattern = os.path.join(self.output_dir, "*.mp4")
        videos = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not videos:
            return None, f"No videos in {self.output_dir} yet."
        latest = videos[-1]
        return latest, f"📁 Loaded latest video: {os.path.basename(latest)}"

    # ---- Optional Background Worker Loop ----
    def start_worker(self, engine):
        """Start a background worker thread that continuously processes queued jobs."""
        with self._lock:
            self._engine = engine
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_worker.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="QueueWorker")
            self._worker_thread.start()
            logger.info("Queuing service worker thread started.")

    def stop_worker(self):
        """Signal worker thread to stop."""
        self._stop_worker.set()
        self._job_available.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def _worker_loop(self):
        """Internal worker loop picking up jobs from the queue."""
        while not self._stop_worker.is_set():
            job_to_run = None
            with self._lock:
                if self._waiting_queue and self._engine is not None:
                    next_id = self._waiting_queue[0]
                    job_to_run = self._jobs.get(next_id)

            if job_to_run is None:
                # Wait for next job or stop signal
                self._job_available.wait(timeout=1.0)
                self._job_available.clear()
                continue

            jid = job_to_run.id
            self.set_job_status(jid, STATUS_RUNNING)

            # Check if cancelled before starting
            if self.is_cancelled(jid):
                self.set_job_status(jid, STATUS_CANCELLED)
                continue

            try:
                def on_progress(frac, desc):
                    self.update_progress(jid, frac, desc)

                def check_cancel():
                    return self.is_cancelled(jid)

                out_path, status_msg = self._engine.generate_video(
                    prompt=job_to_run.prompt,
                    input_image_start=job_to_run.image_path,
                    input_audio=job_to_run.audio_path,
                    seed=job_to_run.seed,
                    duration_dropdown=job_to_run.duration,
                    resolution_dropdown=job_to_run.resolution,
                    aspect_ratio_dropdown=job_to_run.aspect_ratio,
                    match_audio_dur=job_to_run.match_audio_dur,
                    job_id=jid,
                    progress_callback=on_progress,
                    check_cancelled=check_cancel,
                )

                if self.is_cancelled(jid):
                    self.set_job_status(jid, STATUS_CANCELLED)
                elif out_path:
                    self.set_result(jid, out_path)
                else:
                    self.set_job_status(jid, STATUS_FAILED, error=status_msg)

            except Exception as e:
                logger.exception("Error executing job %s: %s", jid, e)
                self.set_job_status(jid, STATUS_FAILED, error=str(e))


# Global singleton instance
queue_service = QueueService()
