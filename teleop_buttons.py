"""
teleop_buttons.py

Click-to-pick, click-to-place teleop for recording fold demonstrations -
with NO KEYBOARD REQUIRED. Everything is done with mouse clicks, because
OpenCV windows on macOS often don't receive keystrokes reliably.

WORKFLOW
--------
1. Click the PICK point on the shirt (green dot appears).
2. Click the PLACE point (red dot + arrow appears).
3. Click the green [CONFIRM STEP] button at the bottom.
   -> the step is saved to disk immediately, and the points clear.
4. Repeat for each fold step.
5. Click the blue [DONE - SAVE & QUIT] button when finished.

The [REDO] button clears the current unsaved points if you misclick.
The [UNDO LAST] button deletes the most recently saved step.

Keyboard shortcuts still work IF your system delivers them (space/r/q),
but you never need them.

AUTO-SAVE
---------
demo.json is rewritten after every confirmed step, so even if the window
is force-closed or the process is killed, your recorded steps survive.

USAGE
-----
    python3 teleop_buttons.py --source shirt.png --out demo_001
    python3 teleop_buttons.py --source 0 --out demo_001
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# EDIT THIS to drive a real robot. Called once per confirmed step.
# ---------------------------------------------------------------------------
def execute_pick_place(pick_xy, place_xy, frame) -> None:
    print(f"    [robot] pick={pick_xy} -> place={place_xy} "
          f"(no robot connected; placeholder)")


@dataclass
class Step:
    step_index: int
    image_file: str
    pick_xy: Tuple[int, int]
    place_xy: Tuple[int, int]
    timestamp: float


# Button bar layout
BAR_H = 60
BTN_H = 40
BTN_Y = 10


class Button:
    def __init__(self, label, x, w, color):
        self.label = label
        self.x = x
        self.w = w
        self.color = color

    def contains(self, x, y):
        return (self.x <= x <= self.x + self.w) and (BTN_Y <= y <= BTN_Y + BTN_H)

    def draw(self, bar, enabled=True):
        color = self.color if enabled else (70, 70, 70)
        cv2.rectangle(bar, (self.x, BTN_Y), (self.x + self.w, BTN_Y + BTN_H),
                      color, -1)
        cv2.rectangle(bar, (self.x, BTN_Y), (self.x + self.w, BTN_Y + BTN_H),
                      (255, 255, 255), 1)
        scale = 0.5
        (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX,
                                       scale, 1)
        tx = self.x + (self.w - tw) // 2
        ty = BTN_Y + (BTN_H + th) // 2
        text_color = (255, 255, 255) if enabled else (140, 140, 140)
        cv2.putText(bar, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, text_color, 1, cv2.LINE_AA)


class TeleopSession:
    MIN_SEP_PX = 10   # pick and place closer than this = probably a misclick

    def __init__(self, source, out_dir: str, max_display: int = 1000):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.is_static = False
        if isinstance(source, str) and os.path.isfile(source):
            self.static_frame = cv2.imread(source)
            if self.static_frame is None:
                raise ValueError(
                    f"Could not read image: {source}\n"
                    "If it came from an iPhone it may be HEIC renamed to .png. "
                    "Convert it first:  sips -s format png IN --out OUT.png")
            self.is_static = True
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(int(source))
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {source}")

        self.steps: List[Step] = []
        self.pick_xy: Optional[Tuple[int, int]] = None
        self.place_xy: Optional[Tuple[int, int]] = None
        self.current_frame = None
        self.quit_requested = False
        self.flash_msg = ""
        self.flash_until = 0.0

        self.max_display = max_display
        self.display_scale = 1.0
        self.disp_w = 0
        self.disp_h = 0

        self.window_name = "Teleop"
        self.buttons = {}

    # --- coordinate helpers -------------------------------------------------
    def to_image(self, x, y):
        s = self.display_scale or 1.0
        return (int(round(x / s)), int(round(y / s)))

    def to_display(self, pt):
        s = self.display_scale or 1.0
        return (int(round(pt[0] * s)), int(round(pt[1] * s)))

    def flash(self, msg, secs=2.0):
        self.flash_msg = msg
        self.flash_until = time.time() + secs

    # --- frames -------------------------------------------------------------
    def get_frame(self):
        if self.is_static:
            return self.static_frame.copy()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return f

    def build_buttons(self, width):
        gap = 8
        n = 4
        w = (width - gap * (n + 1)) // n
        labels = [("confirm", "CONFIRM STEP", (40, 160, 40)),
                  ("redo", "REDO", (40, 120, 200)),
                  ("undo", "UNDO LAST", (30, 90, 160)),
                  ("done", "DONE - SAVE & QUIT", (160, 60, 40))]
        self.buttons = {}
        x = gap
        for key, label, color in labels:
            self.buttons[key] = Button(label, x, w, color)
            x += w + gap

    # --- mouse --------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Clicks below the image are on the button bar.
        if y >= self.disp_h:
            by = y - self.disp_h
            for key, btn in self.buttons.items():
                if btn.contains(x, by):
                    self.on_button(key)
                    return
            return

        # Clicks on the image set pick then place.
        pt = self.to_image(x, y)
        if self.pick_xy is None:
            self.pick_xy = pt
            print(f"  pick  = {pt}")
        elif self.place_xy is None:
            dx = pt[0] - self.pick_xy[0]
            dy = pt[1] - self.pick_xy[1]
            if (dx * dx + dy * dy) ** 0.5 < self.MIN_SEP_PX:
                self.flash("Too close to the pick point - ignored")
                return
            self.place_xy = pt
            print(f"  place = {pt}   -> now click [CONFIRM STEP]")
        else:
            # Third click replaces the pick and starts over, which is what
            # people usually intend when they keep clicking.
            self.pick_xy = pt
            self.place_xy = None
            print(f"  pick  = {pt}  (restarted this step)")

    def on_button(self, key):
        if key == "confirm":
            self.confirm_step()
        elif key == "redo":
            self.redo_step()
        elif key == "undo":
            self.undo_last()
        elif key == "done":
            self.quit_requested = True

    # --- actions ------------------------------------------------------------
    def confirm_step(self):
        if self.pick_xy is None or self.place_xy is None:
            self.flash("Click a PICK point and a PLACE point first")
            return
        idx = len(self.steps)
        img_name = f"step_{idx:02d}.png"
        cv2.imwrite(os.path.join(self.out_dir, img_name), self.current_frame)

        execute_pick_place(self.pick_xy, self.place_xy, self.current_frame)

        self.steps.append(Step(idx, img_name, self.pick_xy, self.place_xy,
                               time.time()))
        self.save_demo()          # auto-save after every step
        print(f"  STEP {idx} SAVED  ({len(self.steps)} total)")
        self.flash(f"Step {idx} saved")
        self.pick_xy = None
        self.place_xy = None

    def redo_step(self):
        self.pick_xy = None
        self.place_xy = None
        self.flash("Cleared - click PICK then PLACE again")

    def undo_last(self):
        if not self.steps:
            self.flash("Nothing to undo")
            return
        s = self.steps.pop()
        path = os.path.join(self.out_dir, s.image_file)
        if os.path.exists(path):
            os.remove(path)
        self.save_demo()
        print(f"  undid step {s.step_index}")
        self.flash(f"Removed step {s.step_index}")

    def save_demo(self):
        out = os.path.join(self.out_dir, "demo.json")
        with open(out, "w") as f:
            json.dump([asdict(s) for s in self.steps], f, indent=2)

    # --- rendering ----------------------------------------------------------
    def render(self, frame):
        h, w = frame.shape[:2]
        self.display_scale = min(1.0, self.max_display / float(max(h, w)))
        if self.display_scale < 1.0:
            vis = cv2.resize(frame, (int(w * self.display_scale),
                                     int(h * self.display_scale)))
        else:
            vis = frame.copy()
        self.disp_h, self.disp_w = vis.shape[:2]

        p = self.to_display(self.pick_xy) if self.pick_xy else None
        q = self.to_display(self.place_xy) if self.place_xy else None
        if p and q:
            cv2.arrowedLine(vis, p, q, (255, 255, 0), 2)
        if p:
            cv2.circle(vis, p, 7, (0, 255, 0), -1)
            cv2.putText(vis, "PICK", (p[0] + 10, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if q:
            cv2.circle(vis, q, 7, (0, 0, 255), -1)
            cv2.putText(vis, "PLACE", (q[0] + 10, q[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # status line
        if self.pick_xy is None:
            status = "Click the PICK point"
        elif self.place_xy is None:
            status = "Click the PLACE point"
        else:
            status = "Click [CONFIRM STEP] below"
        cv2.putText(vis, f"Step {len(self.steps) + 1}   |   "
                         f"{len(self.steps)} saved   |   {status}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2, cv2.LINE_AA)

        if time.time() < self.flash_until:
            cv2.putText(vis, self.flash_msg, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # button bar underneath
        bar = np.full((BAR_H, self.disp_w, 3), 35, dtype=np.uint8)
        self.build_buttons(self.disp_w)
        ready = self.pick_xy is not None and self.place_xy is not None
        self.buttons["confirm"].draw(bar, enabled=ready)
        self.buttons["redo"].draw(bar, enabled=ready or self.pick_xy is not None)
        self.buttons["undo"].draw(bar, enabled=len(self.steps) > 0)
        self.buttons["done"].draw(bar, enabled=True)

        return np.vstack([vis, bar])

    # --- main loop ----------------------------------------------------------
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("\n" + "=" * 62)
        print("  Click PICK point, click PLACE point, then click")
        print("  the green [CONFIRM STEP] button at the bottom.")
        print("  No keyboard needed. Finish with [DONE - SAVE & QUIT].")
        print("  Every confirmed step is written to disk immediately.")
        print("=" * 62 + "\n")

        while not self.quit_requested:
            self.current_frame = self.get_frame()
            cv2.imshow(self.window_name, self.render(self.current_frame))

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                self.confirm_step()
            elif key == ord('r'):
                self.redo_step()
            elif key == ord('q'):
                break

            # If the user closed the window with the red X, stop cleanly.
            try:
                if cv2.getWindowProperty(self.window_name,
                                          cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        self.save_demo()
        out = os.path.join(self.out_dir, "demo.json")
        print(f"\nSaved {len(self.steps)} step(s) to {out}")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="Camera index (0) or path to an image file")
    ap.add_argument("--out", default="demo", help="Output directory")
    ap.add_argument("--max-display", type=int, default=1000,
                    help="Longest window edge in px (coords save at full res)")
    args = ap.parse_args()

    TeleopSession(args.source, args.out, args.max_display).run()


if __name__ == "__main__":
    main()
