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

```bash
# Save a reference setup (defaults to a two-camera SO-101 rig: wrist_camera=0, front_top_camera=1)
teleop-setup-align snapshot

# Later, realign to it
teleop-setup-align align

# Override camera indices if they drift across reconnects/reboots
teleop-setup-align snapshot --camera wrist_camera=0 --camera front_top_camera=1

# Tune the overlay opacity
teleop-setup-align align --alpha 0.4
```

Reference frames are saved under `.setup_reference/` in the current
directory.

**Keys while running:**
- `q` — quit
- `+` / `-` (in `align` mode) — adjust ghost overlay opacity

## Status

Early, personal-workflow tool built around a two-camera SO-101 rig. Camera
handling reuses [`lerobot.cameras.opencv`](https://github.com/huggingface/lerobot),
so it depends on the `lerobot` package. Might be a candidate to integrate
into LeRobot and/or LeLab directly later.

## License

Apache-2.0
