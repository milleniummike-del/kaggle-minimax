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

USE_MOCK = os.environ.get("MOCK", "").strip().lower() in ("true", "1", "yes")
print(USE_MOCK)

if USE_MOCK:
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
        queue_job,
        clear_queue,
        get_queue_status,
        JOB_QUEUE,
        JOB_HISTORY,
        stop_generation,
        load_latest_video,
    )
else:
    from backend import (
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
        queue_job,
        clear_queue,
        get_queue_status,
        JOB_QUEUE,
        JOB_HISTORY,
        stop_generation,
        load_latest_video,
    )


def _snapshot_args(args):
    clean = []
    for a in args:
        if isinstance(a, Image.Image):
            clean.append(a.copy())
        elif isinstance(a, list):
            clean.append([item.copy() if isinstance(item, Image.Image) else item for item in a])
        else:
            clean.append(a)
    return tuple(clean)


def format_queue_display():
    status = get_queue_status()
    lines = []
    if status["running"]:
        lines.append(f"▶️ Currently Running: {status['current_label']}")
    else:
        lines.append("⏸️ Engine is idle (no job running).")

    if status["queue_len"] > 0:
        lines.append(f"📋 Queued ({status['queue_len']}):")
        for idx, item in enumerate(status["queue"], 1):
            lines.append(f"  {idx}. {item['label']}")
    else:
        lines.append("📋 Queue: empty.")
    return "\n".join(lines)


def start_generation(*args):
    prompt_text = str(args[0]).strip() if args and args[0] else ""
    label = (prompt_text[:40] + "...") if len(prompt_text) > 40 else (prompt_text or "Video Generation")

    job_id, pos = queue_job(_snapshot_args(args), label=label)
    if pos == 0:
        status = (f"🚀 Started {label} (Job #{job_id}).\n"
                  f"Progress is shown at the top of the page and the video appears here when it is ready. "
                  f"Long clips take a while; reloading the page or a dropped connection does not stop the job.")
    else:
        status = (f"⏳ Queued {label} as Job #{job_id} (position #{pos} in queue).\n"
                  f"It will execute automatically once preceding jobs complete.")
    return gr.skip(), status, format_queue_display()


def queue_all_manifest_pieces(*ui_args):
    pieces = ZIP_PROJECT.get("pieces", [])
    if not pieces:
        return "❌ Load a ZIP project with manifest.json first.", format_queue_display()

    (_, dialogue, language, sound, music, start_img, end_img, _, ref2, ref3, ref_names, _,
     resolution, aspect, duration, steps, sampler, shift, seed, lora1, s1, lora2, s2,
     custom_url, custom_file, s3, auto_trigger, custom_steps, audio2) = ui_args

    queued_count = 0
    for p in pieces:
        prompt_text = p.get("_prompt", "")
        img = None
        img_path = p.get("_image_path")
        if img_path and os.path.isfile(img_path):
            try:
                with Image.open(img_path) as im:
                    img = im.convert("RGB").copy()
            except Exception:
                img = None
        audio_path = p.get("_audio_path")
        label = p.get("_label", f"Piece {p.get('index', queued_count + 1)}")

        piece_args = (
            prompt_text, dialogue, language, sound, music, start_img, end_img, img, ref2, ref3,
            ref_names, audio_path, resolution, aspect, duration, steps, sampler, shift, seed,
            lora1, s1, lora2, s2, custom_url, custom_file, s3, auto_trigger, custom_steps, audio2
        )
        queue_job(_snapshot_args(piece_args), label=label)
        queued_count += 1

    msg = f"✅ Queued all {queued_count} manifest pieces!\nSettings: {duration} | {resolution} | {steps} | {sampler}"
    return msg, format_queue_display()


def queue_selected_manifest_piece(choice, *ui_args):
    if not choice:
        return "❌ Select a manifest piece first.", format_queue_display()
    piece = next((p for p in ZIP_PROJECT.get("pieces", []) if p["_label"] == choice), None)
    if not piece:
        return f"❌ Piece '{choice}' not found in loaded manifest.", format_queue_display()

    (_, dialogue, language, sound, music, start_img, end_img, _, ref2, ref3, ref_names, _,
     resolution, aspect, duration, steps, sampler, shift, seed, lora1, s1, lora2, s2,
     custom_url, custom_file, s3, auto_trigger, custom_steps, audio2) = ui_args

    prompt_text = piece.get("_prompt", "")
    img = None
    img_path = piece.get("_image_path")
    if img_path and os.path.isfile(img_path):
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB").copy()
        except Exception:
            img = None
    audio_path = piece.get("_audio_path")
    label = piece.get("_label", f"Piece {piece.get('index', 1)}")

    piece_args = (
        prompt_text, dialogue, language, sound, music, start_img, end_img, img, ref2, ref3,
        ref_names, audio_path, resolution, aspect, duration, steps, sampler, shift, seed,
        lora1, s1, lora2, s2, custom_url, custom_file, s3, auto_trigger, custom_steps, audio2
    )
    job_id, pos = queue_job(_snapshot_args(piece_args), label=label)
    if pos == 0:
        msg = f"🚀 Started {label} (Job #{job_id})."
    else:
        msg = f"⏳ Queued {label} as Job #{job_id} (position #{pos} in queue)."
    return msg, format_queue_display()


def clear_queue_action():
    count = clear_queue()
    msg = f"🗑️ Cancelled {count} queued job(s)." if count > 0 else "ℹ️ Queue was already empty."
    return msg, format_queue_display()


def stop_generation_action():
    msg = stop_generation()
    return msg, format_queue_display()


def get_created_videos():
    """List all created .mp4 videos from OUTPUT_DIR (and /kaggle/working/outputs)."""
    search_dirs = [OUTPUT_DIR]
    if os.path.isdir("/kaggle/working/outputs") and os.path.realpath("/kaggle/working/outputs") != os.path.realpath(OUTPUT_DIR):
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
                videos.append(p)
    videos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return videos


def format_video_item(video_path):
    name = os.path.basename(video_path)
    try:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        mtime = os.path.getmtime(video_path)
        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        return f"🎬 {name}  ({size_mb:.1f} MB · {date_str})"
    except Exception:
        return f"🎬 {name}"


def get_created_video_choices():
    return [format_video_item(v) for v in get_created_videos()]


def resolve_video_path(selected_label):
    if not selected_label:
        return None
    for v in get_created_videos():
        if format_video_item(v) == selected_label or os.path.basename(v) in selected_label:
            return v
    return None


def play_created_video(selected_choice):
    if not selected_choice:
        return None, "⚠️ Please select a video to play."
    vpath = resolve_video_path(selected_choice)
    if not vpath or not os.path.isfile(vpath):
        return None, f"❌ File not found: {selected_choice}"
    try:
        size_mb = os.path.getsize(vpath) / (1024 * 1024)
        mtime = os.path.getmtime(vpath)
        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        info = f"▶️ **Now Playing:** `{os.path.basename(vpath)}`\n\n📁 **Path:** `{vpath}` &nbsp;|&nbsp; 💾 **Size:** {size_mb:.2f} MB &nbsp;|&nbsp; 🕒 **Created:** {date_str}"
    except Exception:
        info = f"▶️ **Now Playing:** `{os.path.basename(vpath)}`"
    return vpath, info


def get_created_videos_markdown():
    videos = get_created_videos()
    target_dir = "/kaggle/working/outputs" if os.path.isdir("/kaggle/working/outputs") else OUTPUT_DIR
    if not videos:
        return f"ℹ️ *No created videos found in `{target_dir}` yet. Generated videos will appear here.*"
    lines = [
        f"### 📂 Created Videos ({len(videos)} found in `{target_dir}`)",
        "| # | Video File | Size | Created Date |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for i, v in enumerate(videos, 1):
        name = os.path.basename(v)
        try:
            size_mb = os.path.getsize(v) / (1024 * 1024)
            mtime = os.path.getmtime(v)
            date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            lines.append(f"| **{i}** | `{name}` | {size_mb:.1f} MB | {date_str} |")
        except Exception:
            lines.append(f"| **{i}** | `{name}` | — | — |")
    return "\n".join(lines)


def refresh_created_videos():
    choices = get_created_video_choices()
    first = choices[0] if choices else None
    table_md = get_created_videos_markdown()
    return gr.update(choices=choices, value=first), table_md


def poll_job(seen):
    """Every 2 s: live progress while a job runs, updates video and status as jobs complete."""
    new_video = gr.skip()
    new_seen = seen

    if JOB["done"] and JOB["done"] != seen:
        new_seen = JOB["done"]
        if JOB.get("video") and os.path.isfile(JOB["video"]):
            new_video = JOB["video"]

    status_info = get_queue_status()
    q_len = status_info["queue_len"]
    q_suffix = f" · 📋 {q_len} in queue" if q_len > 0 else ""

    history = status_info.get("history", [])
    choices = [f"Job #{h['id']}: {h['label']}" for h in history if h.get("video")]
    choices_update = gr.update(choices=choices, value=choices[-1] if choices else None)

    q_display = format_queue_display()

    if status_info["running"]:
        elapsed = time.time() - PROGRESS["started"] if PROGRESS["started"] else 0.0
        curr = status_info.get("current_label", f"Job #{status_info['current_id']}")
        status_text = f"⏳ {curr}: {PROGRESS['text'] or 'Working...'} · {_fmt_secs(elapsed)}{q_suffix}"
        if new_seen != seen and JOB.get("status"):
            status_text = f"✅ Finished Job #{new_seen}!\n" + status_text
        if new_seen != seen:
            created_choices = get_created_video_choices()
            created_update = gr.update(choices=created_choices, value=created_choices[0] if created_choices else None)
            created_table = get_created_videos_markdown()
            return new_video, status_text, q_display, choices_update, created_update, created_table, new_seen
        return new_video, status_text, q_display, choices_update, gr.skip(), gr.skip(), new_seen

    if new_seen != seen:
        created_choices = get_created_video_choices()
        created_update = gr.update(choices=created_choices, value=created_choices[0] if created_choices else None)
        created_table = get_created_videos_markdown()
        return new_video, JOB.get("status", "Done."), q_display, choices_update, created_update, created_table, new_seen

    return gr.skip(), gr.skip(), q_display, gr.skip(), gr.skip(), gr.skip(), seen


def select_completed_video(choice):
    if not choice:
        return gr.skip()
    match = re.match(r"Job #(\d+):", choice)
    if not match:
        return gr.skip()
    jid = int(match.group(1))
    status_info = get_queue_status()
    for h in status_info.get("history", []):
        if h["id"] == jid and h.get("video") and os.path.isfile(h["video"]):
            return h["video"]
    return gr.skip()


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
</div>
"""
STATUS_HTML = (f'<div class="status-line">⚡ {INT8_LABEL} &nbsp;|&nbsp; 20B DiT tensor-parallel on 2 x T4 '
               f'&nbsp;|&nbsp; {VARIANT_LABEL} '
               f'&nbsp;|&nbsp; 24 fps video + 32 kHz stereo audio</div>')
FOOTER_HTML = ''
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
                    with gr.Row():
                        queue_all_btn = gr.Button("📥 Queue All Manifest Pieces", variant="secondary")
                        queue_piece_btn = gr.Button("➕ Queue Selected Piece", variant="secondary")
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

        with gr.Accordion("🧑‍🎤 Reference Images", open=False):
            with gr.Row():
                ref1 = gr.Image(type="pil", label="ref1", height=200)
                ref2 = gr.Image(type="pil", label="ref2", height=200)
                ref3 = gr.Image(type="pil", label="ref3", height=200)
                ref_names = gr.Textbox(label="Who / what is in each reference (comma separated, optional)", placeholder="the woman in red, the robot dog, the old castle")
            with gr.Row():
                audio1 = gr.Audio(type="filepath", label="🔊 audio1")
                audio2 = gr.Audio(type="filepath", label="🔊 audio2")
            with gr.Row():
                video1 = gr.Video(label="video1", height=200)
  
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
            gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", scale=2)
            stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn", scale=1)
            clear_queue_btn = gr.Button("📋 Clear Queue", size="lg", scale=1)
            clear_btn = gr.Button("🗑️ Clear Inputs", size="lg", elem_id="clear-btn", scale=1)
        queue_box = gr.Textbox(label="📋 Job Queue", lines=3, interactive=False, value=format_queue_display())
        video_out = gr.Video(label="🎥 Generated video (with audio)", height="720px", width="1280px")
        with gr.Row():
            completed_dropdown = gr.Dropdown(label="🎞️ Completed videos in session", choices=[], value=None, scale=3)
            latest_btn = gr.Button("📂 Load latest video (use if disconnected)", scale=2)
        status_out = gr.Textbox(label="ℹ️ Status", lines=6, interactive=False)
        seen_job = gr.State(0)
        poll_timer = gr.Timer(2.0)

        with gr.Accordion("📂 Created Videos (/kaggle/working/outputs)", open=True):
            gr.Markdown(f"All generated videos in `/kaggle/working/outputs` (`{OUTPUT_DIR}`). Select any video from the list to play or view below:")
            with gr.Row():
                created_video_dropdown = gr.Dropdown(
                    label="🎬 Select Created Video to Play",
                    choices=get_created_video_choices(),
                    value=get_created_video_choices()[0] if get_created_video_choices() else None,
                    scale=4,
                )
                play_created_btn = gr.Button("▶️ Play Video", variant="primary", scale=1)
                refresh_created_btn = gr.Button("🔄 Refresh List", scale=1)
            created_video_player = gr.Video(label="🎥 Video Player", height=450)
            created_video_info = gr.Markdown("")
            created_videos_table = gr.Markdown(value=get_created_videos_markdown())

        gr.Examples(EXAMPLES, inputs=[prompt, dialogue, language, sound, music], label="✨ Example prompts")
        gr.HTML(FOOTER_HTML)

        lora1.change(lora_info, lora1, [s1, lora1_info])
        lora2.change(lora_info, lora2, [s2, lora2_info])
        # Loading the ZIP automatically loads the first piece's prompt, cover image and WAV.
        load_zip_btn.click(load_manifest_zip, inputs=manifest_zip,
                           outputs=[manifest_piece, prompt, ref1, audio1, manifest_status], queue=False)
        # Changing the dropdown immediately refreshes the attached assets; the button remains as a manual reload.
        manifest_piece.change(load_manifest_piece, inputs=manifest_piece,
                              outputs=[prompt, ref1, audio1, manifest_status], queue=False)
        load_piece_btn.click(load_manifest_piece, inputs=manifest_piece,
                             outputs=[prompt, ref1, audio1, manifest_status], queue=False)

        gen_inputs = [prompt, dialogue, language, sound, music, start_img, end_img, ref1, ref2, ref3, ref_names, audio1,
                      resolution, aspect, duration, steps, sampler, shift, seed, lora1, s1, lora2, s2, custom_url,
                      custom_file, s3, auto_trigger, custom_steps, audio2]

        queue_all_btn.click(queue_all_manifest_pieces, inputs=gen_inputs,
                            outputs=[manifest_status, queue_box], queue=False)
        queue_piece_btn.click(queue_selected_manifest_piece, inputs=[manifest_piece] + gen_inputs,
                              outputs=[manifest_status, queue_box], queue=False)

        gen_btn.click(
            start_generation,
            inputs=gen_inputs,
            outputs=[video_out, status_out, queue_box], queue=False)
        poll_timer.tick(poll_job, inputs=seen_job,
                        outputs=[video_out, status_out, queue_box, completed_dropdown, created_video_dropdown, created_videos_table, seen_job], queue=False,
                        show_progress="hidden")
        stop_btn.click(stop_generation_action, outputs=[status_out, queue_box], queue=False)
        clear_queue_btn.click(clear_queue_action, outputs=[status_out, queue_box], queue=False)
        latest_btn.click(load_latest_video, outputs=[video_out, status_out], queue=False)
        completed_dropdown.change(select_completed_video, inputs=completed_dropdown, outputs=video_out, queue=False)
        created_video_dropdown.change(play_created_video, inputs=created_video_dropdown, outputs=[created_video_player, created_video_info], queue=False)
        play_created_btn.click(play_created_video, inputs=created_video_dropdown, outputs=[created_video_player, created_video_info], queue=False)
        refresh_created_btn.click(refresh_created_videos, outputs=[created_video_dropdown, created_videos_table], queue=False)
        clear_btn.click(lambda: (None, "Ready.", "", "", "", "", None, None, None, None, None, "", None, -1),
                        outputs=[video_out, status_out, prompt, dialogue, sound, music, start_img, end_img, ref1, ref2, ref3,
                                 ref_names, audio1, seed], queue=False)

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
    q_len = len(JOB_QUEUE)
    text = PROGRESS.get("text", "")
    if q_len > 0 and PROGRESS.get("active"):
        text = f"{text} (Queue: {q_len})"
    return JSONResponse({**PROGRESS, "text": text, "elapsed": elapsed, "queue_len": q_len}, headers={"Cache-Control": "no-store"})


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
    allowed_dirs = [OUTPUT_DIR, ZIP_PROJECT_DIR]
    if os.path.isdir("/kaggle/working/outputs") and "/kaggle/working/outputs" not in allowed_dirs:
        allowed_dirs.append("/kaggle/working/outputs")
    demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, show_error=True,
                prevent_thread_lock=True, quiet=True, ssr_mode=False, max_threads=8,
                allowed_paths=allowed_dirs)

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
