"""Safe Background Service Launcher for MiniMax H3 Turbo in Jupyter / Kaggle.

Avoids 'OSError: Background processes not supported' by spawning a detached
background process and monitoring startup until the health endpoint is ready.

Usage in a Kaggle / Jupyter notebook cell:
    !python start_service.py

Then in the next cell:
    !python frontend.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("BACKEND_PORT", 8000))
HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
HEALTH_URL = f"http://{HOST}:{PORT}/health"
LOG_FILE = os.environ.get("BACKEND_LOG", "/tmp/backend_service.log")


def is_service_alive() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "AIQ-Launcher"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "ready"
    except Exception:
        pass
    return False


def main():
    print("=" * 70)
    print("🔍 Checking MiniMax H3 Backend Service status...")
    if is_service_alive():
        print(f"✅ Backend Service is ALREADY running on http://{HOST}:{PORT}!")
        print("⚡ You can launch or restart frontend.py anytime without reloading models.")
        print("=" * 70)
        return

    print(f"🚀 Spawning Backend Service on http://{HOST}:{PORT} in background...")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(LOG_FILE)), exist_ok=True)
    except Exception:
        pass

    log_fd = open(LOG_FILE, "a")
    log_fd.write(f"\n--- Launcher started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_fd.flush()

    args = [sys.executable, "backend_service.py", "--host", HOST, "--port", str(PORT)]
    if "--mock" in sys.argv or os.environ.get("MOCK", "").strip().lower() in ("true", "1", "yes"):
        args.append("--mock")

    # Detached process spawn:
    # On POSIX (Linux/Kaggle), start_new_session=True creates a new process session
    # so the backend survives notebook cell completion.
    kwargs = {}
    if hasattr(os, "setsid"):
        kwargs["start_new_session"] = True
    elif sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008  # DETACHED_PROCESS

    proc = subprocess.Popen(args, stdout=log_fd, stderr=subprocess.STDOUT, **kwargs)
    print(f"📌 Process spawned with PID: {proc.pid}")
    print(f"📄 Logs are writing to: {LOG_FILE}")
    print("⏳ Loading video models into GPU VRAM (streaming startup logs)...")
    print("-" * 70)
    sys.stdout.flush()

    # Read log file and stream to console while waiting for health check
    log_reader = open(LOG_FILE, "r")
    log_reader.seek(0, os.SEEK_END)

    start_time = time.time()
    max_wait = 360  # up to 6 minutes for large model loading
    while time.time() - start_time < max_wait:
        line = log_reader.readline()
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            time.sleep(0.5)

        # Check if process terminated prematurely
        if proc.poll() is not None:
            rest = log_reader.read()
            if rest:
                sys.stdout.write(rest)
            print(f"\n❌ Backend service process terminated unexpectedly with code {proc.returncode}!")
            print(f"Check {LOG_FILE} for details.")
            sys.exit(1)

        # Check if health check succeeds
        if is_service_alive():
            print("\n" + "=" * 70)
            print(f"🎉 MiniMax H3 Backend Service is ONLINE and READY! (http://{HOST}:{PORT})")
            print("⚡ Models are resident in memory.")
            print("👉 In the next notebook cell, run: !python frontend.py")
            print("   You can stop, edit, and restart frontend.py anytime in ~1s without reloading models!")
            print("=" * 70 + "\n")
            sys.stdout.flush()
            return

    print(f"⚠️ Service did not report ready within {max_wait}s. Check {LOG_FILE} for details.")


if __name__ == "__main__":
    main()
