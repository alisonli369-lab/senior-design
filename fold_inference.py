"""
fold_inference.py

Load the Cloud Folding LP0LP1 checkpoint (fold-LPLP.pth) and predict a
pick point and a place point for one folding action from an input image.

The model outputs 2 spatial heatmaps at the input resolution:
    channel 0 -> pick  point probability
    channel 1 -> place point probability
We take the argmax of each to get a pixel coordinate.

USAGE
-----
    python fold_inference.py --model fold-LPLP.pth --image shirt.png
    python fold_inference.py --model fold-LPLP.pth --image shirt.png --vis out.png

The model expects 480x640 (HxW) RGB input; other sizes are resized on the
way in and the predicted coordinates are scaled back to your original image
resolution, so the printed coordinates always refer to your input image.

NOTE ON DOMAIN GAP
------------------
These weights were trained on a specific setup: a blue crewneck T-shirt on a
bright pink silicone mat, viewed by an overhead Realsense at a fixed height.
On a different shirt/surface/camera the predictions will likely be poor until
you fine-tune on your own images. Treat this as a starting checkpoint, not a
drop-in solution.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Allow importing the local package folder (foldpkg/) that sits next to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from foldpkg.model import KeypointsGauss  # noqa: E402


# The model was trained at this resolution (config.py in hulk-keypoints).
IMG_HEIGHT = 480
IMG_WIDTH = 640


def load_model(ckpt_path, num_keypoints=2, device="cpu"):
    """Build KeypointsGauss and load a Cloud Folding checkpoint into it."""
    model = KeypointsGauss(num_keypoints=num_keypoints,
                           img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess(image_bgr):
    """HxWx3 uint8 BGR (OpenCV order) -> normalized 1x3x480x640 float tensor.

    Returns the tensor plus the original (H, W) so predictions can be
    scaled back to the caller's image coordinates.
    """
    import cv2

    orig_h, orig_w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (IMG_WIDTH, IMG_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w)


def heatmap_argmax(heatmap):
    """2D array -> (x, y) pixel of the maximum, plus that max value."""
    idx = int(np.argmax(heatmap))
    y, x = np.unravel_index(idx, heatmap.shape)
    return (int(x), int(y)), float(heatmap[y, x])


def predict(model, image_bgr, device="cpu"):
    """Run one forward pass. Returns a dict with pick/place points and heatmaps."""
    tensor, (orig_h, orig_w) = preprocess(image_bgr)
    with torch.no_grad():
        heatmaps = model(tensor.to(device))          # 1 x K x 480 x 640
    heatmaps = heatmaps[0].cpu().numpy()

    # Scale factors to map model-resolution coords back to the input image.
    sx = orig_w / float(IMG_WIDTH)
    sy = orig_h / float(IMG_HEIGHT)

    out = {"heatmaps": heatmaps}
    names = ["pick", "place"]
    for i in range(heatmaps.shape[0]):
        (x, y), conf = heatmap_argmax(heatmaps[i])
        key = names[i] if i < len(names) else f"kp{i}"
        out[key] = (int(round(x * sx)), int(round(y * sy)))
        out[f"{key}_confidence"] = conf
    return out


def visualize(image_bgr, result, out_path):
    """Draw the predicted pick (green) and place (red) points and save."""
    import cv2

    vis = image_bgr.copy()
    if "pick" in result:
        cv2.circle(vis, result["pick"], 10, (0, 255, 0), -1)
        cv2.putText(vis, f"PICK {result['pick_confidence']:.2f}",
                    (result["pick"][0] + 12, result["pick"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if "place" in result:
        cv2.circle(vis, result["place"], 10, (0, 0, 255), -1)
        cv2.putText(vis, f"PLACE {result['place_confidence']:.2f}",
                    (result["place"][0] + 12, result["place"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if "pick" in result and "place" in result:
        cv2.arrowedLine(vis, result["pick"], result["place"], (255, 255, 0), 2)
    cv2.imwrite(out_path, vis)
    print(f"Visualization written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to fold-LPLP.pth")
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument("--num-keypoints", type=int, default=2,
                    help="2 for fold-LPLP, 1 for flatten-LPAP, 3 for flatten-KP")
    ap.add_argument("--vis", default=None, help="Optional path to save a visualization")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    import cv2
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    model = load_model(args.model, num_keypoints=args.num_keypoints,
                       device=args.device)
    result = predict(model, image, device=args.device)

    print(f"Input image: {args.image}  ({image.shape[1]}x{image.shape[0]})")
    for key in ("pick", "place"):
        if key in result:
            print(f"  {key:5s} = {result[key]}  "
                  f"(confidence {result[f'{key}_confidence']:.4f})")

    if args.vis:
        visualize(image, result, args.vis)


if __name__ == "__main__":
    main()
