"""Camera/arm setup alignment utility for repeatable teleop recording sessions.

Physically re-placing a robot arm and cameras after stashing them between
sessions is hard to match by eye. This tool solves it with a ghost overlay:

  1. `snapshot` -- capture one reference frame per camera (the "golden" setup,
     arm included since it's in frame).
  2. `align`    -- show the live feed blended with that reference frame so you
     can nudge the camera/arm until the live view lines up with the ghost.

By default cameras are read straight from the same robot config LeLab and
lerobot already use (`~/.cache/huggingface/lerobot/robots/*.json`), so the
setup you're aligning to always matches what LeLab will actually connect to.

Usage:
    teleop-setup-align snapshot
    teleop-setup-align align
    teleop-setup-align align --alpha 0.4
    teleop-setup-align snapshot --robot so-101
    teleop-setup-align snapshot --camera wrist_camera=0 --camera front_top_camera=1
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Same location LeLab/lerobot write robot+camera setup to (see lelab.utils.config.ROBOTS_PATH).
ROBOTS_PATH = Path("~/.cache/huggingface/lerobot/robots").expanduser()
REFERENCE_DIR = Path(".setup_reference")


def _platform_backend() -> Cv2Backends:
    """Match lelab.record._platform_backend() so we open cameras the same way LeLab does."""
    system = platform.system()
    if system == "Darwin":
        return Cv2Backends.AVFOUNDATION
    if system == "Linux":
        return Cv2Backends.V4L2
    if system == "Windows":
        return Cv2Backends.DSHOW
    return Cv2Backends.ANY


def discover_cameras(robot_name: str | None) -> dict[str, dict]:
    """Read per-camera specs straight from the LeLab/lerobot robot config."""
    if not ROBOTS_PATH.is_dir():
        raise SystemExit(
            f"No robot configs found at {ROBOTS_PATH} (LeLab/lerobot haven't saved a robot setup yet). "
            "Pass --camera NAME=INDEX explicitly, or --robot to point at a different config."
        )

    candidates = sorted(ROBOTS_PATH.glob("*.json"))
    if not candidates:
        raise SystemExit(f"No robot config .json files in {ROBOTS_PATH}. Pass --camera NAME=INDEX explicitly.")

    if robot_name:
        matches = [p for p in candidates if p.stem == robot_name]
        if not matches:
            raise SystemExit(f"No robot config named '{robot_name}' in {ROBOTS_PATH}.")
        config_path = matches[0]
    elif len(candidates) == 1:
        config_path = candidates[0]
    else:
        config_path = max(candidates, key=lambda p: p.stat().st_mtime)
        names = ", ".join(p.stem for p in candidates)
        print(f"Multiple robot configs found ({names}); using most recently modified: {config_path.stem}")
        print("Pass --robot NAME to pick a different one.")

    config = json.loads(config_path.read_text())
    cameras = {}
    for cam in config.get("cameras", []):
        if cam.get("type") != "opencv":
            print(f"Skipping camera '{cam.get('name')}': only type 'opencv' is supported (got '{cam.get('type')}').")
            continue
        cameras[cam["name"]] = {
            "index_or_path": cam["camera_index"],
            "width": cam.get("width"),
            "height": cam.get("height"),
            "fps": cam.get("fps"),
        }

    if not cameras:
        raise SystemExit(f"No opencv cameras found in {config_path}. Pass --camera NAME=INDEX explicitly.")

    print(f"Using cameras from {config_path}: {cameras}")
    return cameras


def parse_cameras(pairs: list[str] | None, robot_name: str | None) -> dict[str, dict]:
    if pairs:
        # Manual override: only the index is known, so leave width/height/fps to the camera's
        # native default rather than guessing values that might not match the real hardware.
        return {
            name: {"index_or_path": int(index), "width": None, "height": None, "fps": None}
            for name, index in (pair.split("=") for pair in pairs)
        }
    return discover_cameras(robot_name)


def open_cameras(cameras: dict[str, dict], warmup_s: int) -> dict[str, OpenCVCamera]:
    backend = _platform_backend()
    handles = {}
    for name, spec in cameras.items():
        cam = OpenCVCamera(OpenCVCameraConfig(backend=backend, warmup_s=warmup_s, **spec))
        cam.connect()
        handles[name] = cam
    return handles


def close_cameras(handles: dict[str, OpenCVCamera]) -> None:
    for cam in handles.values():
        cam.disconnect()


def load_references(camera_names: list[str]):
    """Load saved reference frames (BGR, as written by `snapshot`) for cameras that have one."""
    references = {}
    for name in camera_names:
        ref_path = REFERENCE_DIR / f"{name}.png"
        if ref_path.exists():
            references[name] = cv2.imread(str(ref_path))
    return references


def blend_frame(frame, ref, alpha: float):
    """Blend a live BGR frame with a reference frame as a ghost overlay, if shapes match."""
    if ref is not None and ref.shape == frame.shape:
        return cv2.addWeighted(frame, 1 - alpha, ref, alpha, 0)
    return frame


def mean_luminosity(frame) -> float:
    """Average grayscale brightness of a BGR frame, 0-255."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


# Minimum time between accepted keypresses, so OS key-repeat while a key is held doesn't
# trigger the action multiple times per press. SPACE is a discrete one-shot action so it
# gets a generous window (longer than any normal tap-and-release); +/- adjust an opacity
# slider where a quick repeat while held is actually desirable, so theirs is much shorter.
SAVE_DEBOUNCE_S = 1.5
ALPHA_DEBOUNCE_S = 0.15


def snapshot(cameras: dict[str, dict], warmup_s: int) -> None:
    REFERENCE_DIR.mkdir(exist_ok=True)
    handles = open_cameras(cameras, warmup_s)
    print("Live preview - press SPACE to save a reference frame for every camera, 'q' to quit.")
    try:
        saved = set()
        last_save_t = 0.0
        while True:
            frames = {name: cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR) for name, cam in handles.items()}
            for name, frame in frames.items():
                cv2.imshow(f"{name} (saved)" if name in saved else name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and time.monotonic() - last_save_t > SAVE_DEBOUNCE_S:
                last_save_t = time.monotonic()
                for name, frame in frames.items():
                    out_path = REFERENCE_DIR / f"{name}.png"
                    cv2.imwrite(str(out_path), frame)
                    saved.add(name)
                    print(f"Saved reference: {out_path}")
    finally:
        close_cameras(handles)
        cv2.destroyAllWindows()


def align(cameras: dict[str, dict], alpha: float, warmup_s: int) -> None:
    handles = open_cameras(cameras, warmup_s)
    references = load_references(list(cameras))
    for name in cameras:
        if name not in references:
            print(f"No reference for '{name}' at {REFERENCE_DIR / f'{name}.png'} - run `snapshot` first.")

    print("Move the camera/arm until the live feed lines up with the ghost overlay.")
    print("Keys: 'q' quit, '+'/'-' adjust overlay opacity")
    try:
        last_alpha_t = 0.0
        while True:
            for name, cam in handles.items():
                frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
                frame = blend_frame(frame, references.get(name), alpha)
                cv2.imshow(name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")) and time.monotonic() - last_alpha_t > ALPHA_DEBOUNCE_S:
                last_alpha_t = time.monotonic()
                alpha = min(1.0, alpha + 0.05)
            elif key == ord("-") and time.monotonic() - last_alpha_t > ALPHA_DEBOUNCE_S:
                last_alpha_t = time.monotonic()
                alpha = max(0.0, alpha - 0.05)
    finally:
        close_cameras(handles)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode",
        choices=["snapshot", "align"],
        help="'snapshot' saves the reference setup, 'align' overlays it live",
    )
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=INDEX",
        help="Override a camera, e.g. --camera wrist_camera=0. Repeatable. "
        "Defaults to reading cameras from the LeLab/lerobot robot config.",
    )
    parser.add_argument(
        "--robot",
        metavar="NAME",
        help="Robot config name to read cameras from (matches a file stem under "
        f"{ROBOTS_PATH}). Defaults to the only/most recently modified one.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Initial overlay opacity for 'align' (0-1).")
    parser.add_argument(
        "--warmup-s",
        type=int,
        default=4,
        help="Seconds to let each camera warm up before use (default: 4, some cameras need this to give frames).",
    )
    args = parser.parse_args()

    cameras = parse_cameras(args.camera, args.robot)
    if args.mode == "snapshot":
        snapshot(cameras, args.warmup_s)
    else:
        align(cameras, args.alpha, args.warmup_s)


if __name__ == "__main__":
    main()
