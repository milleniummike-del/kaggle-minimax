"""MiniMax H3 Turbo (INT8 ConvRot) video + audio frontend for Kaggle GPU T4 x2 - AIQUEST Academy.

Gradio Web UI, Cloudflare tunnel integration, and custom FastAPI serving endpoints.
"""
import atexit
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

import gradio as gr
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from PIL import Image

from backend_mock import (
    OUTPUT_DIR,
    ZIP_PROJECT_DIR,
    SERVER_PORT,
    PROGRESS,
    JOB,
    _fmt_secs,
    LANGUAGES,
    RESOLUTIONS,
    ASPECTS,
    DURATIONS,
    STEP_PRESETS,
    DEFAULT_STEPS,
    DEFAULT_SAMPLER,
    LORA_CATALOG,
    LORA_CHOICES,
    NO_LORA,
    REFERENCES_SUPPORTED,
    VARIANT_LABEL,
    INT8_LABEL,
    start_job,
    stop_generation,
    load_latest_video,
)


def start_generation(*args):
    if JOB["running"]:
        return gr.skip(), "⏳ A video is already being generated. It appears here as soon as it is ready."
    start_job(*args)
    return None, ("🚀 Generation started. Progress is shown at the top of the page and the video appears here when it is "
                  "ready. Long clips take a while; reloading the page or a dropped connection does not stop the job.")


def poll_job(seen):
    """Every 2 s: live progress while a job runs, the result once (per page) when it is done."""
    if JOB["running"]:
        elapsed = time.time() - PROGRESS["started"] if PROGRESS["started"] else 0.0
        return gr.skip(), f"⏳ {PROGRESS['text'] or 'Working...'} · {_fmt_secs(elapsed)}", seen
    if JOB["done"] and JOB["done"] != seen:
        return JOB["video"], JOB["status"], JOB["done"]
    return gr.skip(), gr.skip(), seen


# ==== ZIP project / manifest prompt loader ====
ZIP_PROJECT = {"root": None, "pieces": []}


def load_manifest_zip(zip_file):
    """Extract a project ZIP and automatically load its first manifest piece into the UI."""
    empty_choice = gr.update(choices=[], value=None)
    if not zip_file:
        return empty_choice, "", None, None, "❌ Upload a ZIP file first."
    zip_path = zip_file if isinstance(zip_file, str) else getattr(zip_file, "name", None)
    if not zip_path or not os.path.isfile(zip_path):
        return empty_choice, "", None, None, "❌ Could not access the uploaded ZIP."
    project_name = os.path.splitext(os.path.basename(zip_path))[0]
    dest = os.path.join(ZIP_PROJECT_DIR, re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name))
    try:
        # Remove old extraction first so files deleted from a newer ZIP cannot linger.
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(dest, exist_ok=True)
        # Prevent path traversal entries from escaping the project folder.
        root = os.path.realpath(dest)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                target = os.path.realpath(os.path.join(dest, member.filename))
                if target != root and not target.startswith(root + os.sep):
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")
            archive.extractall(dest)
        manifest_path = os.path.join(dest, "manifest.json")
        if not os.path.isfile(manifest_path):
            candidates = glob.glob(os.path.join(dest, "**", "manifest.json"), recursive=True)
            if not candidates:
                raise ValueError("manifest.json was not found in the ZIP.")
            manifest_path = candidates[0]
            dest = os.path.dirname(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        pieces = manifest.get("pieces", [])
        if not pieces:
            raise ValueError("manifest.json contains no 'pieces' entries.")
        choices = []
        clean_pieces = []
        for p in pieces:
            idx = p.get("index", len(clean_pieces) + 1)
            prompt_text = str(p.get("prompt", "")).strip()
            # Manifest entries should point to files inside the extracted project only.
            image_name = p.get("imageFileName") or p.get("image_file")
            audio_name = p.get("audioFileName") or p.get("audio_file") or p.get("audio")
            image_path = os.path.realpath(os.path.join(dest, image_name)) if image_name else None
            audio_path = os.path.realpath(os.path.join(dest, audio_name)) if audio_name else None
            if image_path and not image_path.startswith(os.path.realpath(dest) + os.sep):
                image_path = None
            if audio_path and not audio_path.startswith(os.path.realpath(dest) + os.sep):
                audio_path = None
            if image_path and not os.path.isfile(image_path):
                image_path = None
            if audio_path and not os.path.isfile(audio_path):
                audio_path = None
            label = f"Piece {idx} ({p.get('start', '?')}–{p.get('end', '?')} s)"
            choices.append(label)
            clean_pieces.append({**p, "_label": label, "_prompt": prompt_text,
                                 "_image_path": image_path, "_audio_path": audio_path})
        ZIP_PROJECT.update(root=dest, pieces=clean_pieces)
        summary = [f"✅ Loaded {len(clean_pieces)} pieces from {os.path.basename(manifest_path)}",
                   f"📁 Extracted to: {dest}"]
        for piece in clean_pieces:
            summary.append(f"{piece['_label']}: {piece['_prompt'] or '(empty prompt)'}"
                           + (f" | image: {os.path.basename(piece['_image_path'])}" if piece["_image_path"] else " | ⚠️ no image found")
                           + (f" | audio: {os.path.basename(piece['_audio_path'])}" if piece["_audio_path"] else " | ❌ no WAV found"))
        # Automatically attach the first piece instead of requiring a second click.
        first = load_manifest_piece(choices[0])
        status = "\n".join(summary) + "\n\n" + first[3]
        return gr.update(choices=choices, value=choices[0]), first[0], first[1], first[2], status
    except Exception as e:
        ZIP_PROJECT.update(root=None, pieces=[])
        return empty_choice, "", None, None, f"❌ ZIP/manifest error: {e}"


def load_manifest_piece(choice):
    """Put the selected manifest prompt and cover image into the existing generator inputs."""
    piece = next((p for p in ZIP_PROJECT["pieces"] if p["_label"] == choice), None)
    if not piece:
        return "", None, None, "❌ Load a ZIP and select a piece first."
    prompt_text = piece.get("_prompt", "")
    image = None
    try:
        path = piece.get("_image_path")
        if path:
            with Image.open(path) as im:
                image = im.convert("RGB").copy()
    except Exception as e:
        return prompt_text, None, piece.get("_audio_path"), f"⚠️ Prompt/audio loaded, but cover image could not be opened: {e}"
    loaded = []
    if image is not None:
        loaded.append("cover image")
    if piece.get("_audio_path"):
        loaded.append("WAV audio guide")
    return prompt_text, image, piece.get("_audio_path"), f"✅ Loaded {piece['_label']} prompt" + (" + " + " + ".join(loaded) if loaded else "") + "."


def lora_info(choice):
    if not choice or choice == NO_LORA:
        return gr.update(value=1.0), ""
    entry = LORA_CATALOG[choice]
    trig = f" Trigger: `{entry['trigger']}`" if entry["trigger"] else ""
    return gr.update(value=entry["scale"]), f"ℹ️ {entry['info']}{trig}"


# ==== Gradio UI (AIQUEST Academy branding) ====
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
  <div class="brand-title">🎬 MiniMax H3 Turbo — Video + Audio</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
"""
STATUS_HTML = (f'<div class="status-line">⚡ {INT8_LABEL} &nbsp;|&nbsp; 20B DiT tensor-parallel on 2 x T4 '
               f'&nbsp;|&nbsp; {VARIANT_LABEL} '
               f'&nbsp;|&nbsp; 24 fps video + 32 kHz stereo audio</div>')
FOOTER_HTML = '<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved</div>'

PROGRESS_HEAD = """
<script>
(function () {
  // Progress + result overlay fed by plain HTTP polling: it keeps working when a tunnel drops Gradio's event stream.
  var box = null, fill = null, label = null, hideAt = 0;
  function build() {
    box = document.createElement("div");
    box.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:99999;width:min(560px,92vw);" +
      "background:#1f2937;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);padding:10px 14px;display:none;" +
      "font-family:Inter,sans-serif;color:#fff;font-size:13px";
    label = document.createElement("div");
    label.style.cssText = "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    var track = document.createElement("div");
    track.style.cssText = "height:8px;background:rgba(255,255,255,.18);border-radius:6px;overflow:hidden";
    fill = document.createElement("div");
    fill.style.cssText = "height:100%;width:0;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);transition:width .6s ease";
    track.appendChild(fill); box.appendChild(label); box.appendChild(track); document.body.appendChild(box);
  }
  async function tick() {
    try {
      var r = await fetch("aiq_progress?t=" + Date.now(), {cache: "no-store"});
      var p = await r.json();
      if (!box && document.body) build();
      if (box) {
        if (p.active) {
          box.style.display = "block"; hideAt = 0;
          fill.style.width = Math.round(p.fraction * 100) + "%";
          label.textContent = "⏳ " + p.text + " · " + Math.round(p.elapsed) + "s";
        } else if (box.style.display === "block") {
          if (!hideAt) {
            fill.style.width = "100%";
            label.textContent = (p.video ? "✅ Done in " : "⚠️ Finished after ") + Math.round(p.elapsed) + "s ";
            if (p.video) {
              var link = document.createElement("a");
              link.href = "aiq_video?t=" + Date.now(); link.target = "_blank";
              link.style.cssText = "color:#c7d2fe;font-weight:600";
              link.textContent = "· ▶ open / download the video";
              label.appendChild(link);
            } else if (p.message) { label.textContent += "· " + p.message; }
            hideAt = Date.now() + 60000;
          } else if (Date.now() > hideAt) { box.style.display = "none"; }
        }
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }
  tick();
})();
</script>
"""

EXAMPLES = [
    ["A street musician plays a glowing violin on a rainy neon-lit Tokyo night, the camera slowly circles him as "
     "puddles reflect pink and blue signs and people with umbrellas pass by.",
     "", "English", "Rain on pavement, distant traffic, the violin melody echoing between buildings.", ""],
    ["A young woman in a cozy cafe looks up from her laptop, smiles warmly at the camera and leans forward.",
     "You made it! I saved you the best seat by the window.", "English",
     "Quiet cafe chatter, an espresso machine hissing, cups clinking.", "Soft lo-fi jazz."],
    ["A red fox runs across a snowy ridge at sunrise, powder spraying from every stride as a long-lens camera pans with it.",
     "", "English", "Fast paw impacts in powder, cold wind, distant crows.", "Low sustained cellos rising slowly."],
]


def build_ui():
    print("🖥️ Building the UI...")
    sys.stdout.flush()
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, head=PROGRESS_HEAD, title="MiniMax H3 Turbo | AIQUEST Academy") as demo:
        gr.HTML(BRAND_HTML)
        gr.HTML(STATUS_HTML)
        gr.Markdown("💡 **Tips:** describe subject, action, camera and light. Put exact speech in *Spoken line* for lip-synced "
                    "dialogue. 480p + 5 s is the sweet spot on T4; 360p is the fastest preview. Time per step grows with the "
                    "square of the length × resolution: 480p 10 s ≈ 3x and 15 s ≈ 6x a 5 s clip; 720p 15 s takes ~45 min. "
                    "Long / 720p clips switch to a slower low-VRAM mode automatically.")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="🎬 Scene / action", lines=4,
                                    placeholder="A cinematic shot of ... (or paste a full H3-structured prompt)")
                with gr.Accordion("📦 ZIP project / manifest prompts", open=True):
                    manifest_zip = gr.File(label="Upload project ZIP (must contain manifest.json)", file_types=[".zip"], type="filepath")
                    with gr.Row():
                        load_zip_btn = gr.Button("📂 Extract ZIP + Load Manifest")
                        manifest_piece = gr.Dropdown(label="Manifest piece", choices=[], value=None, scale=2)
                        load_piece_btn = gr.Button("➡️ Load prompt + image")
                    manifest_status = gr.Textbox(label="Manifest status", lines=5, interactive=False)
                    gr.Markdown("The WAV is passed to MiniMax H3 as an audio guide (`audio_guide`), alongside the scene prompt and cover image. It is not merely muxed into the final MP4.")
                with gr.Row():
                    dialogue = gr.Textbox(label="🗣️ Spoken line (optional)", lines=1, scale=3,
                                          placeholder="Exact words the character says")
                    language = gr.Dropdown(LANGUAGES, value="English", label="🌐 Language", scale=1)
                with gr.Row():
                    sound = gr.Textbox(label="🔊 Sound effects / ambience (optional)", lines=1)
                    music = gr.Textbox(label="🎵 Background music (optional)", lines=1)
            with gr.Column(scale=2):
                with gr.Row():
                    resolution = gr.Dropdown(list(RESOLUTIONS), value="480p (recommended)", label="📐 Resolution")
                    aspect = gr.Dropdown(list(ASPECTS), value="Auto (match image)", label="📏 Aspect ratio")
                duration = gr.Dropdown(list(DURATIONS), value="5 s", label="⏱️ Duration")
                with gr.Row():
                    steps = gr.Dropdown(list(STEP_PRESETS), value=next(k for k, v in STEP_PRESETS.items() if v == DEFAULT_STEPS),
                                        label="💎 Quality / speed",
                                        info="More steps = sharper detail; time grows linearly")
                    seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")

        with gr.Accordion("🖼️ Image → Video (start / end frame)", open=False):
            with gr.Row():
                start_img = gr.Image(type="pil", label="Start frame (optional)", height=220)
                end_img = gr.Image(type="pil", label="End frame (optional)", height=220)
            gr.Markdown("*Start only = animate the image. Start + end = first/last-frame interpolation. "
                        "Describe the motion in the prompt; the image fixes the look.*")

        with gr.Accordion("🧑‍🎤 Reference → Video (characters, objects, voice)", open=False):
            with gr.Row():
                ref1 = gr.Image(type="pil", label="Reference 1", height=200)
                ref2 = gr.Image(type="pil", label="Reference 2", height=200)
                ref3 = gr.Image(type="pil", label="Reference 3", height=200)
                ref4 = gr.Video(label="Reference 4", height=200)
            with gr.Row():
                voice = gr.Audio(type="filepath", label="🎙️ Voice reference (optional, 2-15 s)")
                project_audio = gr.Audio(type="filepath", label="🔊 WAV audio guide (auto-loaded from selected manifest piece; optional)")
            with gr.Row():
                ref_names = gr.Textbox(label="Who / what is in each reference (comma separated, optional)",
                                       placeholder="the woman in red, the robot dog, the old castle")

            gr.Markdown("*References keep identity/appearance but place them in a new scene. A voice reference clones "
                        "the voice of Reference 1 for the spoken line.*")

        with gr.Accordion("🎨 LoRAs (style / motion add-ons)", open=False):
            with gr.Row():
                lora1 = gr.Dropdown(LORA_CHOICES, value=NO_LORA, label="LoRA 1")
                s1 = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Strength 1")
            lora1_info = gr.Markdown("")
            with gr.Row():
                lora2 = gr.Dropdown(LORA_CHOICES, value=NO_LORA, label="LoRA 2")
                s2 = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Strength 2")
            lora2_info = gr.Markdown("")
            with gr.Row():
                custom_url = gr.Textbox(label="Custom LoRA URL (Hugging Face .safetensors link)", scale=3,
                                        placeholder="https://huggingface.co/user/repo/resolve/main/my_h3_lora.safetensors")
                custom_file = gr.File(label="...or upload a .safetensors", file_types=[".safetensors"], scale=2)
                s3 = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Custom strength", scale=1)
            auto_trigger = gr.Checkbox(value=True, label="Automatically add each LoRA's trigger words to the prompt")

        with gr.Accordion("⚙️ Advanced", open=False):
            with gr.Row():
                sampler = gr.Dropdown(["euler", "res_multistep"], value=DEFAULT_SAMPLER, label="Sampler",
                                      info="euler: what PDMD / DMAD were run with; res_multistep: crisper audio with the fused turbo")
                shift = gr.Slider(1.0, 16.0, value=12.0, step=0.5, label="Flow shift (video)")
                custom_steps = gr.Slider(0, 12, value=0, step=1, label="Custom steps (0 = use the preset)")

        with gr.Row():
            gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn")
            clear_btn = gr.Button("🗑️ Clear", size="lg", elem_id="clear-btn")
        video_out = gr.Video(label="🎥 Generated video (with audio)", height="720px", width="1280px")
        latest_btn = gr.Button("📂 Load latest video (use if the UI lost its connection)")
        status_out = gr.Textbox(label="ℹ️ Status", lines=6, interactive=False)
        seen_job = gr.State(0)
        poll_timer = gr.Timer(2.0)

        gr.Examples(EXAMPLES, inputs=[prompt, dialogue, language, sound, music], label="✨ Example prompts")
        gr.HTML(FOOTER_HTML)

        lora1.change(lora_info, lora1, [s1, lora1_info])
        lora2.change(lora_info, lora2, [s2, lora2_info])
        # Loading the ZIP automatically loads the first piece's prompt, cover image and WAV.
        load_zip_btn.click(load_manifest_zip, inputs=manifest_zip,
                           outputs=[manifest_piece, prompt, start_img, project_audio, manifest_status], queue=False)
        # Changing the dropdown immediately refreshes the attached assets; the button remains as a manual reload.
        manifest_piece.change(load_manifest_piece, inputs=manifest_piece,
                              outputs=[prompt, start_img, project_audio, manifest_status], queue=False)
        load_piece_btn.click(load_manifest_piece, inputs=manifest_piece,
                             outputs=[prompt, start_img, project_audio, manifest_status], queue=False)
        gen_btn.click(
            start_generation,
            inputs=[prompt, dialogue, language, sound, music, start_img, end_img, ref1, ref2, ref3, ref_names, voice,
                    resolution, aspect, duration, steps, sampler, shift, seed, lora1, s1, lora2, s2, custom_url,
                    custom_file, s3, auto_trigger, custom_steps, project_audio],
            outputs=[video_out, status_out], queue=False)
        poll_timer.tick(poll_job, inputs=seen_job, outputs=[video_out, status_out, seen_job], queue=False,
                        show_progress="hidden")
        stop_btn.click(stop_generation, outputs=status_out, queue=False)
        latest_btn.click(load_latest_video, outputs=[video_out, status_out], queue=False)
        clear_btn.click(lambda: (None, "Ready.", "", "", "", "", None, None, None, None, None, "", None, -1),
                        outputs=[video_out, status_out, prompt, dialogue, sound, music, start_img, end_img, ref1, ref2, ref3,
                                 ref_names, voice, seed], queue=False)

    return demo


demo = build_ui()


# ==== Cloudflare quick tunnel: a second public link in case the Gradio share link fails ====
# Started without preexec_fn: that would force a real fork() of this process (Python otherwise uses vfork), and a
# fork has to copy every pinned page at once (see _dont_fork). Step 2 stops leftover tunnels before each start.
def start_cloudflare_tunnel(port):
    found = {"url": None}
    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        atexit.register(tunnel.kill)
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {e}")
        return found

    def _watch():
        for line in tunnel.stdout:  # keep draining so cloudflared never blocks on a full pipe
            match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)  # skip cloudflared's API host
            if match and not found["url"]:
                found["url"] = match.group(0)

    threading.Thread(target=_watch, daemon=True).start()
    return found


def _progress_route():
    end = PROGRESS["finished"] or time.time()
    elapsed = end - PROGRESS["started"] if PROGRESS["started"] else 0.0
    return JSONResponse({**PROGRESS, "elapsed": elapsed}, headers={"Cache-Control": "no-store"})


def _video_route():
    """The newest video as a plain file (plays in the browser, with a download option): no Gradio involved."""
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return JSONResponse({"error": "no video yet"}, status_code=404)
    name = os.path.basename(videos[-1])
    return FileResponse(videos[-1], media_type="video/mp4", headers={"Content-Disposition": f'inline; filename="{name}"',
                                                                      "Cache-Control": "no-store"})


def main():
    print("🌐 Starting the public links (Cloudflare + Gradio share)...")
    sys.stdout.flush()
    cf = start_cloudflare_tunnel(SERVER_PORT)
    demo.queue(max_size=20)
    demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, show_error=True,
                prevent_thread_lock=True, quiet=True, ssr_mode=False, max_threads=8,
                allowed_paths=[OUTPUT_DIR, ZIP_PROJECT_DIR])

    for _path, _route in (("/aiq_progress", _progress_route), ("/aiq_video", _video_route)):
        demo.app.router.routes.insert(0, APIRoute(_path, _route, methods=["GET"]))  # ahead of Gradio's routes

    for _ in range(40):  # give the tunnel up to ~20 s to report its URL
        if cf["url"]:
            break
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print("🎬 MiniMax H3 Turbo is LIVE!")
    print("=" * 70)
    print(f"🌐 Cloudflare tunnel: {cf['url'] or '(still connecting, check again in a moment)'}")
    print(f"🔗 Gradio share:      {getattr(demo, 'share_url', None) or '(unavailable, use the Cloudflare link)'}")
    print(f"💾 Videos are saved to {OUTPUT_DIR} (Kaggle Output panel)")
    print(f"🎞️ Latest video, without the UI: <link>/aiq_video (e.g. {cf['url'] or 'https://...trycloudflare.com'}/aiq_video)")
    print("=" * 70 + "\n")
    sys.stdout.flush()
    while True:  # keep the server process alive; generation logs stream to the notebook cell
        time.sleep(3600)


if __name__ == "__main__":
    main()
