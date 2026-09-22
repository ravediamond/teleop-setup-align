# teleop-setup-align

A small utility to physically realign a robot arm and cameras to the same
setup every time you record teleop data — no more eyeballing it after you've
stashed and unstashed your rig.

## The problem

If you stash your robot arm and cameras between recording sessions (e.g.
using [LeLab](https://github.com/huggingface/leLab) or
[LeRobot](https://github.com/huggingface/lerobot) to teleoperate and record),
getting the camera framing and arm position back to *exactly* the same spot
by eye is unreliable — and inconsistent camera framing / arm placement across
episodes hurts policy training.

## How it works

Since the arm sits within the camera's field of view, a single reference
image per camera captures *both* the camera framing and the arm pose at once.

1. **`snapshot`** — connect to your cameras, press SPACE to save one
   reference frame per camera as the "golden" setup.
2. **`align`** — reconnect next session and see the live feed blended with
   that reference as a semi-transparent ghost overlay. Physically nudge the
   camera and arm until the live view lines up with the ghost, then start
   recording.

## Install

```bash
pip install teleop-setup-align
# or from source
git clone https://github.com/ravediamond/teleop-setup-align
cd teleop-setup-align
pip install -e .
```

## Usage

By default, cameras are read straight from the same robot config LeLab and
lerobot already write to (`~/.cache/huggingface/lerobot/robots/*.json`), so
the setup you're aligning to always matches what LeLab will actually connect
to — no separate camera list to keep in sync by hand.

```bash
# Save a reference setup (cameras auto-discovered from your LeLab/lerobot robot config)
teleop-setup-align snapshot

# Later, realign to it
teleop-setup-align align

# Multiple robot configs saved? pick one by name
teleop-setup-align snapshot --robot so-101

# Override camera indices if they drift across reconnects/reboots, or if you're
# not using LeLab/lerobot's config at all
teleop-setup-align snapshot --camera wrist_camera=0 --camera front_top_camera=1

# Tune the overlay opacity
teleop-setup-align align --alpha 0.4
```

Reference frames are saved under `.setup_reference/` in the current
directory.

**Keys while running:**
- `q` — quit
- `+` / `-` (in `align` mode) — adjust ghost overlay opacity

## Web UI

A minimal browser version — one page, a live ghost-overlay feed per camera, an
opacity slider, and a Snapshot button. No separate windows or keyboard focus
issues. Prototyped as a rough sketch of what this could look like as a LeLab
page: it mirrors LeLab's own `/camera-feed/{cam_key}` MJPEG-streaming pattern
(`lelab.server` / `lelab.record`), so the pieces (a per-camera MJPEG endpoint,
a snapshot action) would map onto LeLab's FastAPI backend with little change.

```bash
pip install "teleop-setup-align[web]"
teleop-setup-align-web
# open http://127.0.0.1:8420
```

## Status

Early, personal-workflow tool built around a two-camera SO-101 rig. Camera
handling reuses [`lerobot.cameras.opencv`](https://github.com/huggingface/lerobot),
so it depends on the `lerobot` package. Might be a candidate to integrate
into LeRobot and/or LeLab directly later.

## License

Apache-2.0
