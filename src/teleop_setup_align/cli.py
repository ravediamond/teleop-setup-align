"""Camera/arm setup alignment utility for repeatable teleop recording sessions.

Physically re-placing a robot arm and cameras after stashing them between
sessions is hard to match by eye. This tool solves it with a ghost overlay:

  1. `snapshot` -- capture one reference frame per camera (the "golden" setup,
     arm included since it's in frame).
  2. `align`    -- show the live feed blended with that reference frame so you
     can nudge the camera/arm until the live view lines up with the ghost.

Usage:
    teleop-setup-align snapshot
    teleop-setup-align align
    teleop-setup-align align --alpha 0.4
    teleop-setup-align snapshot --camera wrist_camera=0 --camera front_top_camera=1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

DEFAULT_CAMERAS = {
    "wrist_camera": 0,
    "front_top_camera": 1,
}
REFERENCE_DIR = Path(".setup_reference")


def parse_cameras(pairs: list[str] | None) -> dict[str, int]:
    if not pairs:
        return DEFAULT_CAMERAS
    cameras = {}
    for pair in pairs:
        name, index = pair.split("=")
        cameras[name] = int(index)
    return cameras


def open_cameras(cameras: dict[str, int]) -> dict[str, OpenCVCamera]:
    handles = {}
    for name, index in cameras.items():
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480))
        cam.connect()
        handles[name] = cam
    return handles


def close_cameras(handles: dict[str, OpenCVCamera]) -> None:
    for cam in handles.values():
        cam.disconnect()


def snapshot(cameras: dict[str, int]) -> None:
    REFERENCE_DIR.mkdir(exist_ok=True)
    handles = open_cameras(cameras)
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


def align(cameras: dict[str, int], alpha: float) -> None:
    handles = open_cameras(cameras)
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
        help="Override a camera, e.g. --camera wrist_camera=0. Repeatable. Defaults to a two-camera SO-101 setup.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Initial overlay opacity for 'align' (0-1).")
    args = parser.parse_args()

    cameras = parse_cameras(args.camera)
    if args.mode == "snapshot":
        snapshot(cameras)
    else:
        align(cameras, args.alpha)


if __name__ == "__main__":
    main()
