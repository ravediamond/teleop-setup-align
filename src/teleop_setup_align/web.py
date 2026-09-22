"""Minimal web UI for teleop-setup-align, prototyping what a LeLab page could look like.

Instead of separate OpenCV windows and keyboard shortcuts, this serves one browser
page with a live MJPEG feed per camera (ghost overlay blended in server-side, same
as `align`), an opacity slider, and a Snapshot button -- mirrors LeLab's own
`/camera-feed/{cam_key}` MJPEG pattern (see lelab.server / lelab.record) so it would
drop into that app's frontend/backend split with minimal changes.

Run: teleop-setup-align-web
"""

from __future__ import annotations

import argparse
import time

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .cli import (
    REFERENCE_DIR,
    blend_frame,
    close_cameras,
    load_references,
    open_cameras,
    parse_cameras,
)

app = FastAPI(title="teleop-setup-align")

# Single shared session: one browser tab driving one physical rig, same assumption the CLI makes.
state: dict = {"handles": {}, "references": {}, "alpha": 0.5}


def start(cameras: dict[str, dict], warmup_s: int, alpha: float) -> None:
    REFERENCE_DIR.mkdir(exist_ok=True)
    state["handles"] = open_cameras(cameras, warmup_s)
    state["references"] = load_references(list(cameras))
    state["alpha"] = alpha


def stop() -> None:
    close_cameras(state["handles"])


def _mjpeg_frames(name: str):
    cam = state["handles"][name]
    while True:
        frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
        frame = blend_frame(frame, state["references"].get(name), state["alpha"])
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 15)


@app.get("/feed/{name}")
def feed(name: str):
    if name not in state["handles"]:
        return JSONResponse({"error": f"unknown camera '{name}'"}, status_code=404)
    return StreamingResponse(_mjpeg_frames(name), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/snapshot")
def snapshot():
    saved = []
    for name, cam in state["handles"].items():
        frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
        out_path = REFERENCE_DIR / f"{name}.png"
        cv2.imwrite(str(out_path), frame)
        state["references"][name] = frame
        saved.append(name)
    return {"saved": saved}


@app.post("/alpha/{value}")
def set_alpha(value: float):
    state["alpha"] = max(0.0, min(1.0, value))
    return {"alpha": state["alpha"]}


@app.get("/", response_class=HTMLResponse)
def index():
    feeds = "".join(
        f'<div class="cam"><h3>{name}</h3><img src="/feed/{name}"></div>' for name in state["handles"]
    )
    return f"""
<!doctype html>
<title>teleop-setup-align</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  .cams {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .cam img {{ max-width: 100%; width: 480px; border: 1px solid #444; }}
  .controls {{ margin-top: 16px; display: flex; align-items: center; gap: 12px; }}
  button {{ padding: 8px 16px; font-size: 14px; cursor: pointer; }}
  #status {{ opacity: 0.7; }}
</style>
<h2>teleop-setup-align</h2>
<p>Nudge the arm/cameras until the live feed lines up with the ghost overlay, then Snapshot to save a new reference.</p>
<div class="cams">{feeds}</div>
<div class="controls">
  <button onclick="snap()">Snapshot</button>
  <label>Ghost opacity: <input id="alpha" type="range" min="0" max="1" step="0.05" value="{state["alpha"]}"
    oninput="setAlpha(this.value)"></label>
  <span id="status"></span>
</div>
<script>
  async function snap() {{
    const r = await fetch("/snapshot", {{ method: "POST" }});
    const j = await r.json();
    document.getElementById("status").textContent = "Saved: " + j.saved.join(", ");
  }}
  async function setAlpha(v) {{
    await fetch("/alpha/" + v, {{ method: "POST" }});
  }}
</script>
"""


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", metavar="NAME", help="Robot config name to read cameras from.")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=INDEX",
        help="Override a camera, e.g. --camera wrist_camera=0. Repeatable.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Initial overlay opacity (0-1).")
    parser.add_argument("--warmup-s", type=int, default=4, help="Seconds to let each camera warm up.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    cameras = parse_cameras(args.camera, args.robot)
    start(cameras, args.warmup_s, args.alpha)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        stop()


if __name__ == "__main__":
    main()
