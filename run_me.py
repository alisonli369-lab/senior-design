"""
run_me.py  -  the no-terminal version.

Put this file next to fold_inference.py and the foldpkg/ folder, edit the
two settings below, then just Run it (VS Code: the play button; IDLE/Thonny:
Run > Run Module; or double-click it if .py files are set to open with Python).

No command line arguments needed.
"""

import os
import sys

# ---------------------------------------------------------------------------
# EDIT THESE TWO LINES
# ---------------------------------------------------------------------------
MODEL_FILE = "fold-LPLP.pth"   # the checkpoint you downloaded from Box
IMAGE_FILE = "shirt.png"        # a photo of your flattened shirt

# Which checkpoint are you running?
#   fold-LPLP.pth    -> 2   (pick + place)
#   flatten-LPAP.pth -> 1   (pick only)
#   flatten-KP.pth   -> 3   (collar, sleeves, base corners)
NUM_KEYPOINTS = 2

SAVE_VISUALIZATION = True       # writes prediction_output.png with dots drawn on
# ---------------------------------------------------------------------------


# Run everything relative to this file's folder, so it works no matter how
# you launch it (double-click, IDE run button, etc).
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def main():
    # --- friendly dependency check -----------------------------------------
    try:
        import cv2
    except ImportError:
        print("MISSING: opencv-python")
        print("Install it, then run this again:")
        print("    pip install opencv-python")
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        print("MISSING: torch")
        print("Install it, then run this again:")
        print("    pip install torch")
        return

    # --- friendly file check -----------------------------------------------
    if not os.path.exists(MODEL_FILE):
        print(f"Can't find the model file: {MODEL_FILE}")
        print(f"Looked in: {HERE}")
        print("Files that ARE in this folder:")
        for f in sorted(os.listdir(HERE)):
            print("   ", f)
        return

    if not os.path.exists(IMAGE_FILE):
        print(f"Can't find the image: {IMAGE_FILE}")
        print(f"Looked in: {HERE}")
        print("Images that ARE in this folder:")
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        found = [f for f in sorted(os.listdir(HERE)) if f.lower().endswith(exts)]
        for f in found:
            print("   ", f)
        if not found:
            print("    (none - put a photo in this folder first)")
        return

    if not os.path.isdir(os.path.join(HERE, "foldpkg")):
        print("Can't find the 'foldpkg' folder next to this script.")
        print(f"Looked in: {HERE}")
        print("Make sure foldpkg/ (with model.py, resnet.py, resnet_dilated.py,")
        print("__init__.py inside) is in the same folder as this file.")
        return

    # --- run ----------------------------------------------------------------
    from fold_inference import load_model, predict, visualize

    print(f"Loading {MODEL_FILE} ...")
    model = load_model(MODEL_FILE, num_keypoints=NUM_KEYPOINTS, device="cpu")

    image = cv2.imread(IMAGE_FILE)
    print(f"Read {IMAGE_FILE}  ({image.shape[1]} x {image.shape[0]} pixels)")

    result = predict(model, image, device="cpu")

    print("\n--- PREDICTION ---")
    for key in ("pick", "place"):
        if key in result:
            x, y = result[key]
            conf = result[f"{key}_confidence"]
            print(f"  {key.upper():5s}  x={x:4d}  y={y:4d}   confidence={conf:.4f}")

    if SAVE_VISUALIZATION:
        out = "prediction_output.png"
        visualize(image, result, out)

    print("\nReminder: these weights were trained on a blue shirt on a bright")
    print("pink mat. Low confidence on your own photos is expected until you")
    print("fine-tune on your own setup.")


if __name__ == "__main__":
    main()
    # Keep the window open if it was double-clicked from a file browser.
    try:
        input("\nPress Enter to close...")
    except EOFError:
        pass
