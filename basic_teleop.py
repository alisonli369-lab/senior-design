"""
basic_teleop.py

A hardware-agnostic pick-and-place teleop GUI for recording human fold
demonstrations, in the spirit of pyreach/tools/basic_teleop.py from the
Cloud Folding project (https://github.com/ryanhoque/cloudfolding).

Unlike the original, this version has NO dependency on PyReach/Reach -
it works with any camera (a webcam, an Intel Realsense via cv2, or a
folder of pre-captured images) and just records the (pick, place) pixel
coordinates a human clicks. You plug in your own robot-control function
to actually execute the action; this script's job is purely to capture
clean demonstration data for algorithms like ASM or LP0LP1.

WORKFLOW
--------
1. A live camera frame (or a static image) is shown in a window.
2. Click once to mark the PICK point (shown in green).
3. Click again to mark the PLACE point (shown in red), completing one
   fold "step".
4. Press SPACE to confirm and execute the step (calls your
   `execute_pick_place` callback) and save it to the demo log.
5. Press 'r' to redo the current step (clears both points).
6. Press 'n' to start a new step without executing (e.g. if you're just
   recording, not driving a real robot yet).
7. Press 'q' to quit and write out the full demonstration to disk.

The saved demo is a JSON file: a list of steps, each with the frame
image (saved as a .png alongside the JSON) and the pick/place pixel
coordinates. This is exactly the format an ASM or LP0LP1 pipeline wants:
one demonstrated fold sequence made of ordered (pick, place) pairs.

USAGE
-----
    python basic_teleop.py --source 0 --out demo_001
    python basic_teleop.py --source path/to/image.png --out demo_001

Then hook up your robot by editing `execute_pick_place()` below.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# EDIT THIS: plug in your actual robot control here.
# --------------------------------------------------------------------------
def execute_pick_place(pick_xy: Tuple[int, int], place_xy: Tuple[int, int],
                        frame: np.ndarray) -> None:
    """Called once per confirmed step. Replace the body with a call into
    your own arm's API (e.g. compute pixel->world transform, move to
    pick_xy, close gripper, move to place_xy, open gripper).

    frame is the RGB image the points were clicked on, in case your
    pixel->world transform needs the depth frame or camera intrinsics
    captured at the same moment (grab those from your camera driver
    alongside `frame` if needed).
    """
    print(f"[execute_pick_place] pick={pick_xy} place={place_xy} "
          f"(no robot connected - this is a no-op placeholder)")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Step:
    step_index: int
    image_file: str
    pick_xy: Tuple[int, int]
    place_xy: Tuple[int, int]
    timestamp: float


class TeleopSession:
    def __init__(self, source, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.is_static_image = False
        if isinstance(source, str) and os.path.isfile(source):
            self.static_frame = cv2.imread(source)
            if self.static_frame is None:
                raise ValueError(f"Could not read image file: {source}")
            self.is_static_image = True
            self.cap = None
        else:
            # Treat as a camera index (e.g. 0 for the default webcam).
            cam_index = int(source)
            self.cap = cv2.VideoCapture(cam_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera index {cam_index}")

        self.steps = []
        self.pick_xy: Optional[Tuple[int, int]] = None
        self.place_xy: Optional[Tuple[int, int]] = None
        self.current_frame = None
        self.window_name = "Teleop - click PICK then PLACE"

    def get_frame(self) -> np.ndarray:
        if self.is_static_image:
            return self.static_frame.copy()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.pick_xy is None:
            self.pick_xy = (x, y)
            print(f"Pick point set: {self.pick_xy}")
        elif self.place_xy is None:
            self.place_xy = (x, y)
            print(f"Place point set: {self.place_xy}")
        else:
            print("Both points already set - press SPACE to confirm, "
                  "or 'r' to redo before clicking again.")

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        if self.pick_xy is not None:
            cv2.circle(vis, self.pick_xy, 8, (0, 255, 0), -1)
            cv2.putText(vis, "PICK", (self.pick_xy[0] + 10, self.pick_xy[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.place_xy is not None:
            cv2.circle(vis, self.place_xy, 8, (0, 0, 255), -1)
            cv2.putText(vis, "PLACE", (self.place_xy[0] + 10, self.place_xy[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.pick_xy is not None and self.place_xy is not None:
            cv2.arrowedLine(vis, self.pick_xy, self.place_xy, (255, 255, 0), 2)
        hud = ("SPACE=confirm+execute  |  r=redo step  |  "
               "n=skip (no exec)  |  q=quit & save")
        cv2.putText(vis, hud, (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Step {len(self.steps) + 1}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return vis

    def confirm_step(self, execute: bool):
        if self.pick_xy is None or self.place_xy is None:
            print("Need both a pick and a place point before confirming.")
            return
        step_idx = len(self.steps)
        image_file = os.path.join(self.out_dir, f"step_{step_idx:02d}.png")
        cv2.imwrite(image_file, self.current_frame)

        if execute:
            execute_pick_place(self.pick_xy, self.place_xy, self.current_frame)

        self.steps.append(Step(
            step_index=step_idx,
            image_file=os.path.basename(image_file),
            pick_xy=self.pick_xy,
            place_xy=self.place_xy,
            timestamp=time.time(),
        ))
        print(f"Step {step_idx} saved "
              f"({'executed' if execute else 'recorded only'}).")
        self.pick_xy = None
        self.place_xy = None

    def redo_step(self):
        self.pick_xy = None
        self.place_xy = None
        print("Cleared current step - click pick/place again.")

    def save_demo(self):
        out_path = os.path.join(self.out_dir, "demo.json")
        with open(out_path, "w") as f:
            json.dump([asdict(s) for s in self.steps], f, indent=2)
        print(f"\nSaved {len(self.steps)} step(s) to {out_path}")

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("Teleop session started. Click PICK, then PLACE, "
              "then press SPACE.")
        while True:
            self.current_frame = self.get_frame()
            vis = self.draw_overlay(self.current_frame)
            cv2.imshow(self.window_name, vis)

            key = cv2.waitKey(1 if not self.is_static_image else 30) & 0xFF
            if key == ord(' '):
                self.confirm_step(execute=True)
            elif key == ord('n'):
                self.confirm_step(execute=False)
            elif key == ord('r'):
                self.redo_step()
            elif key == ord('q'):
                break

        self.save_demo()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0",
                         help="Camera index (e.g. 0) or path to a static "
                              "image file to click on.")
    parser.add_argument("--out", default="demo",
                         help="Output directory for the saved demo "
                              "(images + demo.json).")
    args = parser.parse_args()

    session = TeleopSession(args.source, args.out)
    session.run()


if __name__ == "__main__":
    main()
