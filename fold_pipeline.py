"""
fold_pipeline.py

The whole thing end to end:

    capture an image  ->  find the shirt and plan the folds  ->  move the arm

It joins shirt_vision.py (the CV half, which works) to the Niryo Ned2 (the
half that still needs proving on real hardware).

SAFETY - READ THIS
------------------
Defaults to a DRY RUN. It prints every pose it would move to and never
touches the robot. You must pass --live to actually move the arm, and it
will ask you to confirm first. Keep the emergency stop within reach.

BEFORE YOU RUN IT LIVE
----------------------
1. pip3 install pyniryo
2. In Niryo Studio, note the robot's IP address.
3. Create and calibrate a WORKSPACE in Niryo Studio - teach it the four
   corners of the flat area the shirt sits on. Pass its name with
   --workspace.
4. Mount the camera looking straight down, and crop/frame it so the image
   covers the SAME rectangle as the calibrated workspace. Any mismatch
   between what the camera sees and what the workspace covers turns into a
   positioning error the robot cannot know about.
5. Attach the gripper and confirm update_tool() detects it.

USAGE
-----
    # plan only, from a photo - no robot needed
    python3 fold_pipeline.py --image shirt.png

    # plan from the webcam
    python3 fold_pipeline.py --camera 0

    # actually fold
    python3 fold_pipeline.py --image shirt.png --live --ip 10.10.10.10 \
        --workspace shirt_ws

RE-ANALYSING BETWEEN FOLDS
--------------------------
By default all three folds are planned from the first image. That is the
simple version, and it drifts: after fold 1 the shirt's outline has changed,
so the plan for folds 2 and 3 is based on a shape that no longer exists.
Pass --recapture to re-photograph and re-plan before each fold. That needs a
live camera, and is the more robust approach.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shirt_vision import analyze, draw_analysis   # noqa: E402


# --------------------------------------------------------------------------
# Motion parameters, in metres relative to the workspace surface.
# --------------------------------------------------------------------------
APPROACH_HEIGHT = 0.06   # hover before descending
PICK_HEIGHT = 0.0        # surface level; go slightly negative to press in
PLACE_HEIGHT = 0.012     # release just above the surface, don't drop
ARM_SPEED = 30           # percent - folding wants slow and smooth


class Arm:
    """Thin wrapper so the dry run and the real robot share one code path."""

    def __init__(self, ip, workspace, dry_run=True):
        self.ip = ip
        self.workspace = workspace
        self.dry_run = dry_run
        self.robot = None

    def connect(self):
        if self.dry_run:
            print(f"[dry run] would connect to Ned2 at {self.ip}")
            return
        from pyniryo import NiryoRobot
        print(f"Connecting to {self.ip} ...")
        self.robot = NiryoRobot(self.ip)
        self.robot.calibrate_auto()
        self.robot.update_tool()
        try:
            self.robot.set_arm_max_velocity(ARM_SPEED)
        except AttributeError:
            # Name differs across pyniryo versions; not worth failing over.
            print("  (couldn't set arm velocity - check your pyniryo version)")
        print("Connected and calibrated.")

    def pose(self, x_rel, y_rel, height):
        if self.dry_run:
            return f"pose(x={x_rel:.3f}, y={y_rel:.3f}, h={height:+.3f})"
        return self.robot.get_target_pose_from_rel(
            self.workspace, height, x_rel, y_rel, 0.0)

    def pick_place(self, pick_rel, place_rel, label=""):
        """One grip: approach, descend, grasp, lift, travel, lower, release."""
        pa = self.pose(*pick_rel, APPROACH_HEIGHT)
        pg = self.pose(*pick_rel, PICK_HEIGHT)
        qa = self.pose(*place_rel, APPROACH_HEIGHT)
        qd = self.pose(*place_rel, PLACE_HEIGHT)

        if self.dry_run:
            for step in ("open gripper", f"move {pa}", f"descend {pg}",
                         "close gripper", f"lift {pa}", f"travel {qa}",
                         f"lower {qd}", "open gripper", f"retreat {qa}"):
                print(f"        [dry] {step}")
            return

        r = self.robot
        r.release_with_tool()
        r.move_pose(pa)
        r.move_pose(pg)
        r.grasp_with_tool()
        r.move_pose(pa)
        r.move_pose(qa)
        r.move_pose(qd)
        r.release_with_tool()
        r.move_pose(qa)

    def finish(self):
        if self.dry_run or self.robot is None:
            return
        self.robot.set_learning_mode(True)
        self.robot.close_connection()


def pixel_to_relative(px, py, img_w, img_h):
    """Image pixel -> workspace-relative [0,1] coords.

    Assumes the image frames exactly the calibrated workspace. Clamped,
    because a point just outside would otherwise command the arm somewhere
    it physically cannot reach.
    """
    x = min(max(px / float(img_w), 0.0), 1.0)
    y = min(max(py / float(img_h), 0.0), 1.0)
    return x, y


def grab_frame(cam_index, warmup=8):
    """Take one photo. Discards a few frames so exposure settles first."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_index}")
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise RuntimeError("Camera opened but returned no frames")
    return frame


def plan_and_report(img, args, tag=""):
    """Run the vision half and print what it found."""
    mask, kp, actions = analyze(img, dark_shirt=(True if args.dark_shirt
                                                 else None),
                                side_fraction=args.side_fraction,
                                hem_fraction=args.hem_fraction)
    h, w = img.shape[:2]
    cover = 100.0 * np.count_nonzero(mask) / mask.size
    print(f"  image {w}x{h}, shirt covers {cover:.1f}% of frame")
    if cover < 8 or cover > 75:
        print("  !! that coverage looks wrong - check the annotated image "
              "before going live")

    out = args.out if not tag else args.out.replace(".png", f"_{tag}.png")
    cv2.imwrite(out, draw_analysis(img, mask, kp, actions))
    print(f"  annotated image -> {out}")
    return actions, (w, h)


def run(args):
    arm = Arm(args.ip, args.workspace, dry_run=not args.live)

    # --- source of images -------------------------------------------------
    def get_image():
        if args.camera is not None:
            return grab_frame(args.camera)
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(
                f"Could not read {args.image}. If it came off an iPhone it "
                f"may be HEIC renamed to .png - convert with:\n"
                f"    sips -s format png {args.image} --out fixed.png")
        return img

    print("=" * 64)
    print("PLANNING")
    print("=" * 64)
    img = get_image()
    actions, (w, h) = plan_and_report(img, args)

    for i, act in enumerate(actions, 1):
        print(f"\n  {i}. {act['label']}")
        for j, (pick, place) in enumerate(act["grips"], 1):
            print(f"       grip {j}: {pick} -> {place}")

    if args.recapture and args.camera is None:
        print("\n--recapture needs a live camera (--camera N). "
              "Falling back to a single plan.")
        args.recapture = False

    # --- execute ----------------------------------------------------------
    print("\n" + "=" * 64)
    print("EXECUTING" if args.live else "EXECUTING (DRY RUN - arm will not move)")
    print("=" * 64)

    arm.connect()

    for i in range(len(actions)):
        if args.recapture and i > 0:
            print(f"\n  re-photographing before fold {i + 1} ...")
            img = get_image()
            try:
                actions_now, (w, h) = plan_and_report(img, args, tag=f"step{i+1}")
            except ValueError as e:
                print(f"  couldn't re-analyse ({e}); stopping here rather "
                      f"than folding blind.")
                break
            if i >= len(actions_now):
                print("  re-analysis produced fewer folds; stopping.")
                break
            act = actions_now[i]
        else:
            act = actions[i]

        print(f"\n  --- fold {i + 1}: {act['label']} ---")
        for j, (pick, place) in enumerate(act["grips"], 1):
            pr = pixel_to_relative(pick[0], pick[1], w, h)
            qr = pixel_to_relative(place[0], place[1], w, h)
            print(f"      grip {j}: px{pick} -> px{place}   "
                  f"rel({pr[0]:.3f},{pr[1]:.3f}) -> ({qr[0]:.3f},{qr[1]:.3f})")
            arm.pick_place(pr, qr, act["label"])

    arm.finish()
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a photo of the flattened shirt")
    src.add_argument("--camera", type=int, help="Camera index to capture from")

    ap.add_argument("--out", default="pipeline_analysis.png",
                    help="Where to write the annotated image")
    ap.add_argument("--dark-shirt", action="store_true",
                    help="Force dark-shirt-on-light-background polarity")
    ap.add_argument("--side-fraction", type=float, default=1.0 / 3.0,
                    help="Where the side creases sit (0.333 = fold in thirds)")
    ap.add_argument("--hem-fraction", type=float, default=0.5,
                    help="Where the horizontal crease sits (0.5 = in half)")

    ap.add_argument("--ip", default="10.10.10.10", help="Ned2 IP address")
    ap.add_argument("--workspace", default="shirt_ws",
                    help="Calibrated workspace name from Niryo Studio")
    ap.add_argument("--live", action="store_true",
                    help="ACTUALLY MOVE THE ARM (default is a dry run)")
    ap.add_argument("--recapture", action="store_true",
                    help="Re-photograph and re-plan before each fold "
                         "(needs --camera). More robust than one-shot.")
    args = ap.parse_args()

    if args.live:
        print("*** LIVE MODE - THE ARM WILL MOVE ***")
        print(f"    robot {args.ip}, workspace '{args.workspace}'")
        print("    Keep the emergency stop within reach.")
        try:
            resp = input("    Type 'yes' to continue: ").strip().lower()
        except EOFError:
            resp = ""
        if resp != "yes":
            print("Aborted.")
            return

    run(args)


if __name__ == "__main__":
    main()
