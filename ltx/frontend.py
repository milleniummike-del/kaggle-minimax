"""
Frontend Interface for LTX-2.3 Audio-to-Video Engine.
Contains the Gradio UI components, styling, AIQUEST branding, and event handling.
"""

import os
import sys
import glob
import time
import argparse
import subprocess
import threading
import re
import stat
import urllib.request
from typing import Optional, Tuple, List, Any

# Ensure local imports work regardless of current working directory
ORIG_DIR = os.path.dirname(os.path.abspath(__file__))
if ORIG_DIR not in sys.path:
    sys.path.insert(0, ORIG_DIR)

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False

from queue_service import (
    queue_service,
    STATUS_RUNNING,
    STATUS_FAILED,
    STATUS_CANCELLED,
    OUTPUT_DIR,
)
from video_engine import (
    video_engine,
    AUTO_ASPECT,
    NUM_GPUS,
)

# ---- UI Styling & Brand Components ----
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.btn-row { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.social-btn { display: inline-flex; align-items: center; justify-content: center; min-width: 150px; padding: 10px 18px; border-radius: 10px; font-weight: 700; font-size: 13px; text-decoration: none; color: white; white-space: nowrap; }
.yt-btn  { background: #FF0000; box-shadow: 0 4px 12px rgba(255,0,0,0.3); }
.x-btn   { background: #000000; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
.sup-btn { background: linear-gradient(135deg,#f6d365,#fda085); box-shadow: 0 4px 12px rgba(253,160,133,0.35); }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.status-line { text-align: center; color: #6b7280; font-size: 13px; margin: -8px 0 12px 0; }
.brand-footer { text-align: center; color: #6b7280; font-size: 12px; margin-top: 24px; }
"""

BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">🎤 LTX-2.3 22B Audio-to-Video / Lip Sync</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved</div>
"""


def _get_status_html() -> str:
    gpu_layout = "Gemma + upsampler on GPU 1" if video_engine.ups_on_gpu1 else (
        "Gemma on GPU 1" if video_engine.aux_on_gpu1 else ("Dual GPU" if NUM_GPUS > 1 else "Single GPU")
    )
    return f'<div class="status-line">⚡ GGUF Q4_K_M transformer (FP16) &nbsp;|&nbsp; {gpu_layout} &nbsp;|&nbsp; FP32 VAE</div>'


# ---- Queue & Bridge Functions for Gradio ----
async def register_job_fn(prompt: str) -> str:
    """Register job immediately on click to show in queue display without waiting for worker."""
    job = queue_service.register_job(prompt=prompt)
    return job.id


async def clear_waiting_jobs_fn():
    """Cancel all waiting jobs."""
    queue_service.clear_waiting_jobs()


async def get_queue_status_fn() -> str:
    """Return text status of active queue."""
    return queue_service.get_queue_status_text()


async def get_finished_videos_fn() -> List[str]:
    """Return list of generated video paths."""
    return queue_service.get_finished_videos(limit=12)


def load_latest_video_fn() -> Tuple[Optional[str], str]:
    """Load latest generated video from outputs directory."""
    return queue_service.load_latest_video()


def video_generation_fn(
    prompt: str,
    input_image_start: Optional[str],
    input_audio: Optional[str],
    seed: int,
    duration_dropdown: str,
    resolution_dropdown: str,
    aspect_ratio_dropdown: str,
    match_audio_dur: bool = False,
    job_id: Optional[str] = None,
    progress: Any = None,
) -> Tuple[Optional[str], str]:
    """
    Bridge function called by Gradio's generate button or Gradio client predict(/generate_video).
    Connects Gradio's progress bar to both the QueueService and VideoEngine.
    """
    if progress is None and HAS_GRADIO:
        progress = gr.Progress()

    # If job was not pre-registered (e.g. called directly via API / check.js), register it now
    if not job_id or not queue_service.get_job(job_id):
        job = queue_service.register_job(
            prompt=prompt,
            image_path=input_image_start,
            audio_path=input_audio,
            seed=seed,
            duration=duration_dropdown,
            resolution=resolution_dropdown,
            aspect_ratio=aspect_ratio_dropdown,
            match_audio_dur=match_audio_dur,
            job_id=job_id,
        )
        job_id = job.id

    try:
        queue_service.set_job_status(job_id, STATUS_RUNNING)
        if progress and callable(progress):
            progress(0.0, desc="Starting...")

        def progress_cb(frac: float, desc: str):
            queue_service.update_progress(job_id, frac, desc)
            if progress and callable(progress):
                progress(frac, desc=desc)

        def cancel_cb() -> bool:
            return queue_service.is_cancelled(job_id)

        out_path, status_text = video_engine.generate_video(
            prompt=prompt,
            input_image_start=input_image_start,
            input_audio=input_audio,
            seed=seed,
            duration_dropdown=duration_dropdown,
            resolution_dropdown=resolution_dropdown,
            aspect_ratio_dropdown=aspect_ratio_dropdown,
            match_audio_dur=match_audio_dur,
            job_id=job_id,
            progress_callback=progress_cb,
            check_cancelled=cancel_cb,
        )

        if queue_service.is_cancelled(job_id):
            queue_service.set_job_status(job_id, STATUS_CANCELLED)
            return None, "🛑 Generation stopped by user."

        if out_path:
            queue_service.set_result(job_id, out_path)
            if progress and callable(progress):
                progress(1.0, desc="Done!")
            return out_path, status_text
        else:
            queue_service.set_job_status(job_id, STATUS_FAILED, error=status_text)
            return None, status_text

    except Exception as e:
        queue_service.set_job_status(job_id, STATUS_FAILED, error=str(e))
        return None, f"❌ Error: {str(e)}"

    finally:
        queue_service.remove_job(job_id)


# ---- Create Gradio Interface ----
def create_ui():
    """Build and return the Gradio demo interface."""
    if not HAS_GRADIO:
        raise RuntimeError("Gradio is not installed. Install with: pip install gradio")

    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="LTX-2.3 Audio-to-Video | AIQUEST Academy") as demo:
        gr.HTML(BRAND_HTML)
        gr.HTML(_get_status_html())

        gr.Markdown(
            "💡 **Tip:** Tag your prompt with `[VISUAL] <scene> [SPEECH] <dialogue> [SOUND] <voice description>` for highest accuracy."
        )

        with gr.Column():
            prompt = gr.Textbox(
                label="📝 Prompt",
                lines=3,
                placeholder="[VISUAL] A professional speaker talking directly to camera in a modern studio. [SPEECH] Welcome everyone to our new AI tutorial! [SOUND] Clear expressive voice, studio microphone.",
            )

            with gr.Row():
                input_image_start = gr.Image(type="filepath", label="🖼️ Reference Image (Portrait / Start Frame)")
                input_audio = gr.Audio(type="filepath", label="🎤 Speech / Vocals Audio Track")

            with gr.Row():
                seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
                duration_dropdown = gr.Dropdown(
                    label="⏱️ Duration",
                    choices=[
                        "2 Seconds (49 frames)",
                        "3 Seconds (73 frames)",
                        "5 Seconds (121 frames)",
                        "8 Seconds (193 frames)",
                        "10 Seconds (241 frames)",
                        "15 Seconds (361 frames)",
                        "20 Seconds (481 frames)",
                    ],
                    value="5 Seconds (121 frames)",
                )
                match_audio_dur = gr.Checkbox(
                    label="🎵 Auto-match video duration to uploaded audio file",
                    value=False,
                    info="When checked, snaps video frame count automatically to the exact duration of your audio file.",
                )

            match_audio_dur.change(
                fn=lambda chk: gr.update(interactive=not chk),
                inputs=[match_audio_dur],
                outputs=[duration_dropdown],
            )

            with gr.Row():
                resolution_dropdown = gr.Dropdown(
                    label="📐 Base Resolution",
                    choices=["720p", "540p", "480p"],
                    value="720p",
                )
                aspect_ratio_dropdown = gr.Dropdown(
                    label="📏 Aspect Ratio",
                    choices=[AUTO_ASPECT, "16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                    value=AUTO_ASPECT,
                    info="Auto follows the reference image (16:9 without one)",
                )

            with gr.Row():
                gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_id="gen-btn")
                stop_btn = gr.Button("🛑 Stop", variant="secondary", size="lg", elem_id="stop-btn")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", elem_id="clear-btn")

            job_state = gr.State()
            queue_display = gr.Textbox(label="Current Queue", interactive=False, lines=4)

            video_out = gr.Video(label="🎥 Generated Video")
            latest_btn = gr.Button("📂 Load Latest Video (use if the UI showed an error)", variant="secondary")
            status_out = gr.Textbox(label="ℹ️ Status", interactive=False)
            gallery = gr.Gallery(label="📼 Finished videos (newest first)", columns=3, height="auto", object_fit="contain")

            # Periodic Refresh: Queue & Finished Videos
            timer = gr.Timer(1.0)
            timer.tick(fn=get_queue_status_fn, inputs=None, outputs=queue_display, api_name="get_queue_status")
            timer.tick(fn=get_finished_videos_fn, inputs=None, outputs=gallery, api_name="get_finished_videos")

            latest_btn.click(fn=load_latest_video_fn, outputs=[video_out, status_out])

            # Generation Pipeline:
            # 1. register_job_fn registers job immediately in QueueService (queue=False bypasses concurrency limit)
            # 2. video_generation_fn runs inference (concurrency_limit=1 enforces strict GPU serialization)
            gen_event = gen_btn.click(
                fn=register_job_fn,
                inputs=[prompt],
                outputs=[job_state],
                queue=False,
                show_progress="hidden",
                trigger_mode="multiple",
            ).then(
                fn=video_generation_fn,
                inputs=[
                    prompt,
                    input_image_start,
                    input_audio,
                    seed,
                    duration_dropdown,
                    resolution_dropdown,
                    aspect_ratio_dropdown,
                    match_audio_dur,
                    job_state,
                ],
                outputs=[video_out, status_out],
                concurrency_limit=1,
                show_progress="full",
                api_name="generate_video",
            )

            stop_btn.click(fn=clear_waiting_jobs_fn, inputs=None, outputs=None, cancels=[gen_event], queue=False)
            clear_btn.click(
                fn=lambda: (None, None, "", -1),
                outputs=[video_out, status_out, prompt, seed],
            )

        gr.HTML(FOOTER_HTML)

    return demo


def start_cloudflare_tunnel(port: int = 7860):
    """Start Cloudflare Quick Tunnel for Gradio."""
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
                print(f"* Cloudflare tunnel URL: {match.group(0)}  (use this if the Gradio link does not load)")
                sys.stdout.flush()
                break
        for _ in tunnel.stdout:
            pass

    threading.Thread(target=_watch, daemon=True).start()


def launch_frontend(server_port: int = 7860, share: bool = True, tunnel: bool = True):
    """Launch the Gradio interface."""
    if not HAS_GRADIO:
        raise RuntimeError("Gradio is not installed. Run 'pip install gradio' to start the UI.")

    demo = create_ui()
    if tunnel:
        start_cloudflare_tunnel(server_port)

    print("\nLaunching Gradio Frontend Interface...")
    sys.stdout.flush()

    demo.queue(default_concurrency_limit=1, max_size=128)
    demo.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        share=share,
        inline=False,
        debug=True,
        show_error=True,
        max_threads=4,
        ssr_mode=False,
        allowed_paths=[OUTPUT_DIR],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2.3 Gradio Frontend")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default 7860)")
    parser.add_argument("--share", action="store_true", default=True, help="Enable Gradio public share link")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable Cloudflare quick tunnel")
    args = parser.parse_args()

    launch_frontend(server_port=args.port, share=args.share, tunnel=not args.no_tunnel)
