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
from pathlib import Path

import cv2
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Same location LeLab/lerobot write robot+camera setup to (see lelab.utils.config.ROBOTS_PATH).
ROBOTS_PATH = Path("~/.cache/huggingface/lerobot/robots").expanduser()
REFERENCE_DIR = Path(".setup_reference")


def discover_cameras(robot_name: str | None) -> dict[str, int]:
    """Read camera name -> index straight from the LeLab/lerobot robot config."""
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
        cameras[cam["name"]] = cam["camera_index"]

    if not cameras:
        raise SystemExit(f"No opencv cameras found in {config_path}. Pass --camera NAME=INDEX explicitly.")

    print(f"Using cameras from {config_path}: {cameras}")
    return cameras


def parse_cameras(pairs: list[str] | None, robot_name: str | None) -> dict[str, int]:
    if pairs:
        return {name: int(index) for name, index in (pair.split("=") for pair in pairs)}
    return discover_cameras(robot_name)


def open_cameras(cameras: dict[str, int], warmup_s: int) -> dict[str, OpenCVCamera]:
    handles = {}
    for name, index in cameras.items():
        cam = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480, warmup_s=warmup_s)
        )
        cam.connect()
        handles[name] = cam
    return handles


def close_cameras(handles: dict[str, OpenCVCamera]) -> None:
    for cam in handles.values():
        cam.disconnect()


def snapshot(cameras: dict[str, int], warmup_s: int) -> None:
    REFERENCE_DIR.mkdir(exist_ok=True)
    handles = open_cameras(cameras, warmup_s)
    print("Live preview - press SPACE to save a reference frame for every camera, 'q' to quit.")
    try:
        saved = set()
        while True:
            frames = {name: cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR) for name, cam in handles.items()}
            for name, frame in frames.items():
                cv2.imshow(f"{name} (saved)" if name in saved else name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                for name, frame in frames.items():
                    out_path = REFERENCE_DIR / f"{name}.png"
                    cv2.imwrite(str(out_path), frame)
                    saved.add(name)
                    print(f"Saved reference: {out_path}")
    finally:
        close_cameras(handles)
        cv2.destroyAllWindows()


def align(cameras: dict[str, int], alpha: float, warmup_s: int) -> None:
    handles = open_cameras(cameras, warmup_s)
    references = {}
    for name in cameras:
        ref_path = REFERENCE_DIR / f"{name}.png"
        if not ref_path.exists():
            print(f"No reference for '{name}' at {ref_path} - run `snapshot` first. Showing live only.")
            continue
        references[name] = cv2.imread(str(ref_path))

    print("Move the camera/arm until the live feed lines up with the ghost overlay.")
    print("Keys: 'q' quit, '+'/'-' adjust overlay opacity")
    try:
        while True:
            for name, cam in handles.items():
                frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
                ref = references.get(name)
                if ref is not None and ref.shape == frame.shape:
                    frame = cv2.addWeighted(frame, 1 - alpha, ref, alpha, 0)
                cv2.imshow(name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                alpha = min(1.0, alpha + 0.05)
            elif key == ord("-"):
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
