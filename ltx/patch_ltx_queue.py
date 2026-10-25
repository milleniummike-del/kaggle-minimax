#!/usr/bin/env python3
\"\"\"
Patch an existing LTX-2.3 Wan2GP script to use Gradio's sequential event queue.

Usage:
    python patch_ltx_queue.py /path/to/run_ltx_audio.py
    python patch_ltx_queue.py /path/to/run_ltx_audio.py --output /path/to/run_ltx_audio_queued.py

This preserves the model-loading and generation code. Each Generate click becomes a
separate queued job, and the queue concurrency is limited to one GPU generation.
\"\"\"
import argparse
from pathlib import Path
import sys

OLD_EVENT = '''        gen_event = gen_btn.click(
            fn=Video_Generation,
            inputs=[
                prompt, input_image_start, input_audio, seed, duration_dropdown,
                resolution_dropdown, aspect_ratio_dropdown,
                match_audio_dur
            ],
            outputs=[video_out, status_out],
        )
        stop_btn.click(fn=None, cancels=[gen_event])'''

NEW_EVENT = '''        # Every click is a distinct queued job; submitted input values are captured.
        gen_event = gen_btn.click(
            fn=Video_Generation,
            inputs=[
                prompt, input_image_start, input_audio, seed, duration_dropdown,
                resolution_dropdown, aspect_ratio_dropdown,
                match_audio_dur
            ],
            outputs=[video_out, status_out],
            concurrency_limit=1,
            trigger_mode="multiple",
            show_progress="full",
            api_name="generate_video",
        )
        stop_btn.click(fn=None, cancels=[gen_event])'''

OLD_QUEUE = '''demo.queue()
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, debug=True,
            show_error=True, max_threads=1, ssr_mode=False, allowed_paths=[OUTPUT_DIR])'''

NEW_QUEUE = '''# Run only one model generation at a time; Gradio shows queue position/progress.
demo.queue(default_concurrency_limit=1, max_size=128)
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, debug=True,
            show_error=True, max_threads=1, ssr_mode=False, allowed_paths=[OUTPUT_DIR])'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to your existing LTX-2.3 Python script")
    parser.add_argument("--output", help="Output path (default: create a *_queued.py copy)")
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.is_file():
        sys.exit(f"ERROR: Source script not found: {src}")

    text = src.read_text(encoding="utf-8")
    if OLD_EVENT not in text:
        sys.exit("ERROR: Expected Generate event block not found; source was not changed.")
    if OLD_QUEUE not in text:
        sys.exit("ERROR: Expected demo.queue()/demo.launch() block not found; source was not changed.")

    patched = text.replace(OLD_EVENT, NEW_EVENT, 1).replace(OLD_QUEUE, NEW_QUEUE, 1)
    dst = Path(args.output).expanduser().resolve() if args.output else src.with_name(src.stem + "_queued" + src.suffix)
    dst.write_text(patched, encoding="utf-8")
    print(f"Patched script written to: {dst}")
    print("Each Generate click is queued; generations run one at a time.")
    print("MP4s continue to be saved in OUTPUT_DIR.")

if __name__ == "__main__":
    main()
