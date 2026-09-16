"""
niryo_fold.py

Connects the teleop/ASM pick-and-place pipeline to a Niryo Ned2.

THE KEY IDEA
------------
Ned2's "workspace" feature solves the pixel -> robot-coordinate problem for
you. You teach the robot the 4 corners of a flat rectangular work area once
(in Niryo Studio), and from then on:

    get_target_pose_from_rel(workspace, height_offset, x_rel, y_rel, yaw_rel)

converts a RELATIVE position inside that workspace (both x_rel and y_rel run
0.0 to 1.0) into a full robot pose. So all we have to do is convert our
image pixel (px, py) into those 0-1 relative coordinates, which is just a
division - provided the camera view is cropped to the workspace.

    x_rel = px / image_width
    y_rel = py / image_height

BEFORE RUNNING THIS
-------------------
1. In Niryo Studio, connect to the robot and note its IP address.
2. Create and calibrate a workspace (Niryo Studio -> Workspaces). Name it
   and put that name in WORKSPACE_NAME below. The 4 landmarks must be the
   corners of the flat area your shirt sits on.
3. Mount your camera so its view matches the workspace as closely as
   possible. Any mismatch between "what the camera sees" and "what the
   workspace covers" becomes a positioning error.
4. Attach the gripper and make sure update_tool() detects it.

SAFETY
------
Start with DRY_RUN = True. It prints every pose it *would* move to without
moving the arm. Only set it False once the printed poses look sane and you
have the emergency stop within reach.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# SETTINGS - edit these
# ---------------------------------------------------------------------------
ROBOT_IP = "10.10.10.10"        # from Niryo Studio
WORKSPACE_NAME = "shirt_ws"      # the workspace you calibrated in Studio

DRY_RUN = True                    # True = print poses only, never move the arm

# Height offsets in metres, relative to the workspace surface.
PICK_HEIGHT = 0.0                # how deep to descend when grabbing fabric
                                   # slightly negative presses into the shirt
APPROACH_HEIGHT = 0.06           # hover height above pick/place points
PLACE_HEIGHT = 0.01              # release just above the surface

# The paper's folding primitive: move slowly, grip deep enough to catch
# multiple fabric layers, and lower at the place point instead of dropping.
ARM_SPEED = 30                    # percent; folding wants slow and smooth


def pixel_to_relative(px, py, img_w, img_h):
    """Image pixel -> workspace-relative coords in [0, 1].

    Assumes the image is cropped to exactly the calibrated workspace. If your
    camera sees more than the workspace, crop first or this will be skewed.
    """
    x_rel = px / float(img_w)
    y_rel = py / float(img_h)
    # Clamp, because a click just outside the workspace would otherwise ask
    # the arm to reach somewhere it cannot go.
    x_rel = min(max(x_rel, 0.0), 1.0)
    y_rel = min(max(y_rel, 0.0), 1.0)
    return x_rel, y_rel


class NiryoFolder:
    def __init__(self, ip=ROBOT_IP, workspace=WORKSPACE_NAME, dry_run=DRY_RUN):
        self.ip = ip
        self.workspace = workspace
        self.dry_run = dry_run
        self.robot = None

    def connect(self):
        if self.dry_run:
            print(f"[DRY RUN] Would connect to Ned2 at {self.ip}")
            return
        from pyniryo import NiryoRobot

        print(f"Connecting to Ned2 at {self.ip} ...")
        self.robot = NiryoRobot(self.ip)
        self.robot.calibrate_auto()
        self.robot.update_tool()
        self.robot.set_arm_max_velocity(ARM_SPEED)
        print("Connected and calibrated.")

    def pose_for(self, x_rel, y_rel, height_offset, yaw=0.0):
        """Ask the robot to convert workspace-relative coords into a pose."""
        if self.dry_run:
            return f"<pose x_rel={x_rel:.3f} y_rel={y_rel:.3f} h={height_offset}>"
        return self.robot.get_target_pose_from_rel(
            self.workspace, height_offset, x_rel, y_rel, yaw)

    def fold_step(self, pick_px, place_px, img_w, img_h):
        """Execute one pick-and-place fold action, in the paper's style:
        approach high, descend, grip, lift, travel, lower, release."""
        px_rel = pixel_to_relative(pick_px[0], pick_px[1], img_w, img_h)
        pl_rel = pixel_to_relative(place_px[0], place_px[1], img_w, img_h)

        print(f"  pick  px={pick_px}  -> rel=({px_rel[0]:.3f}, {px_rel[1]:.3f})")
        print(f"  place px={place_px} -> rel=({pl_rel[0]:.3f}, {pl_rel[1]:.3f})")

        pick_approach = self.pose_for(*px_rel, APPROACH_HEIGHT)
        pick_grasp = self.pose_for(*px_rel, PICK_HEIGHT)
        place_approach = self.pose_for(*pl_rel, APPROACH_HEIGHT)
        place_down = self.pose_for(*pl_rel, PLACE_HEIGHT)

        if self.dry_run:
            print(f"    [DRY] open gripper")
            print(f"    [DRY] move to {pick_approach}")
            print(f"    [DRY] descend to {pick_grasp}")
            print(f"    [DRY] close gripper (grab fabric)")
            print(f"    [DRY] lift to {pick_approach}")
            print(f"    [DRY] travel to {place_approach}")
            print(f"    [DRY] lower to {place_down}")
            print(f"    [DRY] open gripper (release)")
            return

        r = self.robot
        r.release_with_tool()               # open
        r.move_pose(pick_approach)
        r.move_pose(pick_grasp)
        r.grasp_with_tool()                  # close on the fabric
        r.move_pose(pick_approach)           # lift
        r.move_pose(place_approach)          # travel across
        r.move_pose(place_down)              # lower, don't drop from height
        r.release_with_tool()                # release
        r.move_pose(place_approach)          # retreat up

    def run_demo(self, demo_path):
        """Replay a recorded demo.json as a fold sequence."""
        with open(demo_path) as f:
            steps = json.load(f)
        if not steps:
            print("demo.json has no steps in it.")
            return

        # Image size comes from the recorded frame, so relative coords match
        # whatever resolution the demo was captured at.
        import cv2
        demo_dir = os.path.dirname(os.path.abspath(demo_path))
        first_img = os.path.join(demo_dir, steps[0]["image_file"])
        img = cv2.imread(first_img)
        if img is None:
            print(f"Could not read {first_img} to determine image size.")
            return
        img_h, img_w = img.shape[:2]
        print(f"Demo images are {img_w} x {img_h}\n")

        self.connect()
        for s in steps:
            print(f"--- Step {s['step_index']} ---")
            self.fold_step(tuple(s["pick_xy"]), tuple(s["place_xy"]),
                           img_w, img_h)

        if not self.dry_run:
            self.robot.set_learning_mode(True)
            self.robot.close_connection()
        print("\nFold sequence complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="demo_001/demo.json",
                    help="Path to a demo.json recorded with the teleop tool")
    ap.add_argument("--ip", default=ROBOT_IP, help="Ned2 IP address")
    ap.add_argument("--workspace", default=WORKSPACE_NAME,
                    help="Name of the calibrated workspace in Niryo Studio")
    ap.add_argument("--live", action="store_true",
                    help="ACTUALLY MOVE THE ARM (default is a dry run)")
    args = ap.parse_args()

    if not os.path.exists(args.demo):
        print(f"Can't find {args.demo}")
        print("Record one first with:  python3 teleop_buttons.py "
              "--source shirt.png --out demo_001")
        sys.exit(1)

    folder = NiryoFolder(ip=args.ip, workspace=args.workspace,
                         dry_run=not args.live)
    if args.live:
        print("*** LIVE MODE - THE ARM WILL MOVE ***")
        print("Keep the emergency stop within reach.")
        input("Press Enter to continue, or Ctrl+C to abort...")

    folder.run_demo(args.demo)


if __name__ == "__main__":
    main()
