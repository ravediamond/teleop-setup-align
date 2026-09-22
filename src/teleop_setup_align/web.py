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
    lighting_match,
    lighting_signature,
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


def _mjpeg_frames(name: str, ghost: bool):
    cam = state["handles"][name]
    while True:
        frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
        if ghost:
            frame = blend_frame(frame, state["references"].get(name), state["alpha"])
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 15)


@app.get("/feed/{name}")
def feed(name: str, ghost: int = 1):
    if name not in state["handles"]:
        return JSONResponse({"error": f"unknown camera '{name}'"}, status_code=404)
    return StreamingResponse(
        _mjpeg_frames(name, ghost=bool(ghost)), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
def status():
    return {"references": {name: name in state["references"] for name in state["handles"]}, "alpha": state["alpha"]}


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


@app.get("/luminosity/{name}")
def luminosity(name: str):
    if name not in state["handles"]:
        return JSONResponse({"error": f"unknown camera '{name}'"}, status_code=404)
    ref = state["references"].get(name)
    if ref is None:
        return {"has_reference": False}
    live_frame = cv2.cvtColor(state["handles"][name].read(), cv2.COLOR_RGB2BGR)
    match = lighting_match(lighting_signature(live_frame), lighting_signature(ref))
    return {"has_reference": True, "match": match}


@app.get("/", response_class=HTMLResponse)
def index():
    camera_names = list(state["handles"])
    cards = "".join(
        f"""
        <div class="bg-white border border-gray-200 rounded-lg shadow-sm">
          <div class="flex items-center justify-between p-4 pb-2">
            <span class="text-xs font-semibold uppercase tracking-wide text-gray-500">{name}</span>
            <span data-badge="{name}" class="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold bg-gray-100 border-gray-300 text-gray-600">no reference</span>
          </div>
          <div class="p-4 pt-2">
            <div class="relative rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
              <img data-feed="{name}" class="w-full aspect-video object-cover" />
            </div>
            <div data-lum-wrap="{name}" class="mt-2 h-8 hidden">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs font-medium text-gray-600">Lighting match</span>
                <span data-lum-pct="{name}" class="text-xs font-semibold text-gray-700"></span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-1.5">
                <div data-lum-bar="{name}" class="h-1.5 rounded-full transition-all duration-300" style="width:0%"></div>
              </div>
            </div>
          </div>
        </div>"""
        for name in camera_names
    )

    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>teleop-setup-align</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gray-50 text-gray-900">
  <div class="max-w-4xl mx-auto p-4">
    <div class="mb-6">
      <h1 class="text-3xl font-bold">teleop-setup-align</h1>
      <p class="text-gray-500 text-sm mt-1">Realign a stashed rig to a saved reference setup</p>
    </div>

    <div class="inline-flex rounded-md border border-gray-200 bg-white p-1 mb-4">
      <button id="tab-1" onclick="setStep(1)"
        class="rounded-sm px-4 py-1.5 text-sm font-medium transition-colors">1&nbsp;&middot;&nbsp;Snapshot</button>
      <button id="tab-2" onclick="setStep(2)"
        class="rounded-sm px-4 py-1.5 text-sm font-medium transition-colors">2&nbsp;&middot;&nbsp;Align</button>
    </div>

    <p id="step-help" class="text-sm text-gray-500 mb-4"></p>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
      {cards}
    </div>

    <div id="panel-1" class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 flex items-center gap-3">
      <button onclick="snap()"
        class="inline-flex items-center justify-center rounded-md text-sm font-medium h-10 px-4 py-2 bg-green-500 text-white hover:bg-green-600 transition-colors">
        Save Snapshot
      </button>
      <span id="snap-status" class="text-sm text-gray-500"></span>
    </div>

    <div id="panel-2" class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 flex items-center gap-4 hidden">
      <label class="text-sm font-medium text-gray-700">Ghost opacity</label>
      <input id="alpha" type="range" min="0" max="1" step="0.05" value="{state["alpha"]}"
        oninput="setAlpha(this.value)" class="w-48 accent-green-500" />
      <span id="alpha-value" class="text-sm text-gray-500 w-10">{int(state["alpha"] * 100)}%</span>
    </div>
  </div>

<script>
  const cameraNames = {camera_names!r};
  let step = 1;

  function setStep(n) {{
    step = n;
    document.getElementById("panel-1").classList.toggle("hidden", n !== 1);
    document.getElementById("panel-2").classList.toggle("hidden", n !== 2);
    for (const [id, active] of [["tab-1", n === 1], ["tab-2", n === 2]]) {{
      const el = document.getElementById(id);
      el.className = "rounded-sm px-4 py-1.5 text-sm font-medium transition-colors " +
        (active ? "bg-green-500 text-white" : "text-gray-500 hover:text-gray-900");
    }}
    document.getElementById("step-help").textContent = n === 1
      ? "Get the arm and cameras where you want them, then save a reference frame per camera."
      : "Nudge the arm/cameras until the live feed lines up with the saved reference (the ghost).";
    for (const name of cameraNames) {{
      document.querySelector(`img[data-feed="${{name}}"]`).src = `/feed/${{name}}?ghost=${{n === 2 ? 1 : 0}}&t=${{Date.now()}}`;
      document.querySelector(`[data-lum-wrap="${{name}}"]`).classList.toggle("hidden", n !== 2);
    }}
  }}

  // Optional: the ghost overlay checks framing, but not whether the room lighting has
  // changed since the reference was taken. Plain mean-brightness doesn't catch that well
  // because auto-exposure actively renormalizes it -- the backend instead scores the worst
  // of brightness/color-balance/sharpness divergence, so e.g. closing curtains (which shifts
  // color temperature and adds noise even if AE keeps the mean looking similar) still shows up.
  async function pollLuminosity() {{
    if (step === 2) {{
      for (const name of cameraNames) {{
        const r = await fetch(`/luminosity/${{name}}`);
        const j = await r.json();
        const bar = document.querySelector(`[data-lum-bar="${{name}}"]`);
        const pctEl = document.querySelector(`[data-lum-pct="${{name}}"]`);
        if (!j.has_reference) {{ bar.style.width = "0%"; pctEl.textContent = "no reference"; continue; }}
        const match = Math.round(j.match);
        const color = match >= 85 ? "bg-green-500" : match >= 60 ? "bg-amber-500" : "bg-red-500";
        bar.style.width = match + "%";
        bar.className = "h-1.5 rounded-full transition-all duration-300 " + color;
        pctEl.textContent = match + "%";
      }}
    }}
    setTimeout(pollLuminosity, 1500);
  }}

  async function refreshStatus() {{
    const r = await fetch("/status");
    const j = await r.json();
    for (const [name, has] of Object.entries(j.references)) {{
      const badge = document.querySelector(`[data-badge="${{name}}"]`);
      badge.textContent = has ? "reference saved" : "no reference";
      badge.className = "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold " +
        (has ? "bg-green-100 border-green-300 text-green-700" : "bg-gray-100 border-gray-300 text-gray-600");
    }}
  }}

  async function snap() {{
    const r = await fetch("/snapshot", {{ method: "POST" }});
    const j = await r.json();
    document.getElementById("snap-status").textContent = "Saved: " + j.saved.join(", ");
    await refreshStatus();
  }}

  async function setAlpha(v) {{
    document.getElementById("alpha-value").textContent = Math.round(v * 100) + "%";
    await fetch("/alpha/" + v, {{ method: "POST" }});
  }}

  setStep(1);
  refreshStatus();
  pollLuminosity();
</script>
</body>
</html>
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
