"""
shirt_vision.py

Find a T-shirt's landmarks in an image and compute where to pick and place
for each fold - using classical computer vision only. No training data, no
GPU, no pretrained model.

WHY THIS INSTEAD OF THE LEARNED MODEL
-------------------------------------
The Cloud Folding paper's own results found that the analytic shape-matching
policy (ASM, IoU 0.69) slightly BEAT the learned pick-place policy
(LP0LP1, IoU 0.68) on the folding subtask. Folding a flattened shirt is a
well-defined geometric problem, so geometry solves it.

PIPELINE
--------
1. segment_shirt()   - isolate the shirt from the background
2. find_keypoints()  - locate collar, shoulders, sleeve tips, hem corners
3. plan_folds()      - turn those landmarks into (pick, place) actions
4. draw_analysis()   - render everything so you can check it by eye

ASSUMPTIONS
-----------
- The shirt is already flattened and lying flat (you place it by hand).
- The camera looks straight down at it.
- The background contrasts with the shirt (a dark mat under a light shirt,
  or vice versa). This is the single biggest factor in whether it works.
- The shirt is roughly upright OR rotated - rotation is handled, the shirt
  is analysed in its own frame and results are mapped back to image pixels.

USAGE
-----
    python3 shirt_vision.py --image shirt.png
    python3 shirt_vision.py --image shirt.png --out analysis.png
    python3 shirt_vision.py --image shirt.png --dark-shirt
"""

import argparse
import math

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. SEGMENTATION
# ---------------------------------------------------------------------------
def segment_shirt(img, dark_shirt=None, min_area_frac=0.02):
    """Return a binary mask of the largest shirt-like blob.

    Uses Otsu thresholding on the saturation+value channels, which handles a
    plain background far more reliably than a hand-tuned colour range. If
    dark_shirt is None the polarity is chosen automatically by assuming the
    shirt occupies less of the frame than the background does.
    """
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Otsu finds the split between shirt and background automatically.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Decide which side of the split is the shirt.
    if dark_shirt is None:
        # The shirt should be the smaller region; if the "white" side covers
        # more than half the frame it is probably background, so invert.
        if np.count_nonzero(mask) > mask.size * 0.5:
            mask = cv2.bitwise_not(mask)
    elif dark_shirt:
        mask = cv2.bitwise_not(mask)

    # Clean up specks and fill small holes (buttons, logos, shadows).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    # Keep only the largest connected component - the shirt.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    if areas.max() < mask.size * min_area_frac:
        return None          # nothing big enough to be a shirt
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. KEYPOINTS
# ---------------------------------------------------------------------------
def _rotate_pts(pts, M):
    ones = np.ones((len(pts), 1), np.float32)
    return np.hstack([pts, ones]) @ M.T


def _orientation_score(rot_pts, n_bins=24):
    """How much does this orientation look like an upright T-shirt?

    A T-shirt laid flat has a distinctive vertical width profile: the sleeves
    make it WIDEST near the top, then it narrows sharply at the armpits and
    stays roughly constant down the torso to the hem.

    So the score is (widest row in the top half) / (typical row in the bottom
    half). Upright scores high. Upside-down scores low, because then the wide
    sleeve region sits at the bottom. Sideways scores near 1, because a shirt
    rotated 90 degrees has a much flatter profile.

    This replaces the old assumption that the long axis is collar-to-hem,
    which breaks on boxy or cropped shirts that are wider than they are tall.
    """
    ys, xs = rot_pts[:, 1], rot_pts[:, 0]
    top, bottom = ys.min(), ys.max()
    height = bottom - top
    if height < 1e-6:
        return -1.0

    edges = np.linspace(top, bottom, n_bins + 1)
    widths = np.zeros(n_bins)
    for i in range(n_bins):
        sel = xs[(ys >= edges[i]) & (ys <= edges[i + 1])]
        widths[i] = (sel.max() - sel.min()) if len(sel) > 1 else 0.0

    valid = widths > 0
    if valid.sum() < n_bins * 0.5:
        return -1.0

    half = n_bins // 2
    top_max = widths[:half].max()
    bottom_vals = widths[half:][widths[half:] > 0]
    if len(bottom_vals) == 0:
        return -1.0
    bottom_med = float(np.median(bottom_vals))
    if bottom_med < 1e-6:
        return -1.0
    return float(top_max / bottom_med)


def _canonical_frame(mask):
    """Find the shirt's own axes so a rotated shirt still works.

    Tries all four right-angle orientations of the minimum-area box and keeps
    whichever one actually looks like an upright T-shirt, rather than assuming
    the longest side is the collar-to-hem axis.
    """
    # CHAIN_APPROX_NONE keeps EVERY boundary pixel. CHAIN_APPROX_SIMPLE
    # collapses straight runs to their endpoints, which leaves most of the
    # horizontal bins in the width profile empty and makes the orientation
    # score unusable.
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0],
              key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), _, base_angle = rect

    pts = cnt.reshape(-1, 2).astype(np.float32)

    best = None
    for extra in (0.0, 90.0, 180.0, 270.0):
        angle = base_angle + extra
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot_pts = _rotate_pts(pts, M)
        score = _orientation_score(rot_pts)
        if best is None or score > best[0]:
            best = (score, angle, M, rot_pts)

    score, angle, M, rot_pts = best
    M_inv = cv2.invertAffineTransform(M)
    return rot_pts, M, M_inv, (cx, cy)


def _to_image(pt, M_inv):
    """Upright-shirt coordinate -> original image pixel."""
    v = np.array([pt[0], pt[1], 1.0], np.float32)
    out = M_inv @ v
    return (int(round(out[0])), int(round(out[1])))


def find_keypoints(mask):
    """Locate the shirt landmarks. Returns a dict in IMAGE pixel coords,
    plus the geometry needed by the fold planner."""
    rot, M, M_inv, center = _canonical_frame(mask)

    xs, ys = rot[:, 0], rot[:, 1]
    top, bottom = ys.min(), ys.max()
    height = bottom - top
    if height < 10:
        return None

    def band(y0, y1):
        """Points whose y falls in a horizontal slice of the shirt."""
        sel = rot[(ys >= y0) & (ys <= y1)]
        return sel if len(sel) else None

    def extreme_point(pts, want_max_x, tol=8.0):
        """Find the extreme-x point, but return the MIDDLE of that edge.

        A sleeve tip is a short vertical edge, so simply taking argmin/argmax
        of x lands on whichever corner happens to come first in the contour -
        which makes the left and right sleeves come out asymmetric. Averaging
        the y of all points at that extreme gives the centre of the edge.
        """
        col = pts[:, 0]
        target = col.max() if want_max_x else col.min()
        on_edge = pts[np.abs(col - target) <= tol]
        if len(on_edge) == 0:
            on_edge = pts[[np.argmax(col) if want_max_x else np.argmin(col)]]
        return np.array([float(target), float(np.median(on_edge[:, 1]))])

    # --- Sleeve tips: the widest points, which occur in the upper body ---
    upper = band(top, top + height * 0.55)
    if upper is None:
        return None
    left_sleeve = extreme_point(upper, want_max_x=False)
    right_sleeve = extreme_point(upper, want_max_x=True)

    # --- Hem corners: extremes of the bottom edge ---
    lower = band(bottom - height * 0.12, bottom)
    if lower is None:
        return None
    hem_left = np.array([float(lower[:, 0].min()), float(bottom)])
    hem_right = np.array([float(lower[:, 0].max()), float(bottom)])
    hem_mid = np.array([(hem_left[0] + hem_right[0]) / 2.0, bottom])

    # --- Torso edges: measured below the sleeves so they aren't included ---
    torso = band(top + height * 0.55, bottom)
    torso_left = float(torso[:, 0].min())
    torso_right = float(torso[:, 0].max())

    # --- Collar: centred on the torso, at the top edge. Using the torso
    # centre rather than the median of the top band keeps it centred even
    # when one shoulder is slightly more visible than the other. ---
    collar = np.array([(torso_left + torso_right) / 2.0, float(top)])

    # --- Shoulder corners: the outer ends of the shoulder line. Together
    # with the hem corners these pin down the four corners of each side
    # panel, which is what actually gets folded over.
    # Take the REAL contour point, not (min_x, top) - the shoulder slopes,
    # so the leftmost point in the band sits lower than the top edge and
    # pairing it with top would put the grip off the fabric entirely. ---
    shoulder_band = band(top, top + height * 0.10)
    shoulder_left = shoulder_band[np.argmin(shoulder_band[:, 0])].copy()
    shoulder_right = shoulder_band[np.argmax(shoulder_band[:, 0])].copy()

    geom = {
        "top": float(top), "bottom": float(bottom), "height": float(height),
        "torso_left": torso_left, "torso_right": torso_right,
        "M_inv": M_inv,
    }
    kp_rot = {
        "collar": collar,
        "shoulder_left": shoulder_left,
        "shoulder_right": shoulder_right,
        "left_sleeve": left_sleeve,
        "right_sleeve": right_sleeve,
        "hem_left": hem_left,
        "hem_right": hem_right,
        "hem_mid": hem_mid,
    }
    kp_img = {k: _to_image(v, M_inv) for k, v in kp_rot.items()}
    return {"image": kp_img, "rot": kp_rot, "geom": geom}


# ---------------------------------------------------------------------------
# 3. FOLD PLANNING
# ---------------------------------------------------------------------------
def plan_folds(kp, side_fraction=1.0 / 3.0, hem_fraction=0.5):
    """Turn landmarks into the three-fold sequence from the standard method.

    A fold is a REFLECTION across a crease line, not a drag of one point. So
    for each fold we work out where the crease is, then mirror the corners of
    the panel being folded across it. That gives the exact place point for
    every grip, and it is what makes the fabric land flat instead of bunching.

    Each fold returns TWO grips (the panel's top and bottom corners), because
    a single grip point pivots the fabric and creases it diagonally. Grabbing
    both ends of the edge keeps the fold square - that is what the hands in
    the tutorial are doing.

      1. left panel folded in   - crease runs vertically at side_fraction
                                  of the torso width from the left edge
      2. right panel folded in  - mirror of step 1
      3. hem folded up          - crease runs horizontally at hem_fraction
                                  of the remaining height

    side_fraction 1/3 reproduces the tutorial's "fold the left third over".
    hem_fraction 0.5 folds the strip in half; use ~0.33 to fold in thirds.
    """
    r = kp["rot"]
    g = kp["geom"]
    M_inv = g["M_inv"]

    torso_w = g["torso_right"] - g["torso_left"]
    top, bottom = g["top"], g["bottom"]

    actions = []

    def mirror_x(pt, crease_x):
        return np.array([2.0 * crease_x - pt[0], pt[1]])

    def mirror_y(pt, crease_y):
        return np.array([pt[0], 2.0 * crease_y - pt[1]])

    def emit(label, grips, crease):
        """grips is a list of (pick, place) in the upright frame."""
        actions.append({
            "label": label,
            "crease": crease,
            "grips": [(_to_image(p, M_inv), _to_image(q, M_inv))
                      for p, q in grips],
        })

    # ---- Fold 1: left panel over, crease a third in from the left ----
    crease1_x = g["torso_left"] + torso_w * side_fraction
    grips1 = [
        (r["shoulder_left"], mirror_x(r["shoulder_left"], crease1_x)),
        (r["hem_left"], mirror_x(r["hem_left"], crease1_x)),
    ]
    emit("fold left third over", grips1,
         ((crease1_x, top), (crease1_x, bottom)))

    # ---- Fold 2: right panel over, crease a third in from the right ----
    crease2_x = g["torso_right"] - torso_w * side_fraction
    grips2 = [
        (r["shoulder_right"], mirror_x(r["shoulder_right"], crease2_x)),
        (r["hem_right"], mirror_x(r["hem_right"], crease2_x)),
    ]
    emit("fold right third over", grips2,
         ((crease2_x, top), (crease2_x, bottom)))

    # ---- Fold 3: hem up. After folds 1 and 2 the shirt is a narrow strip
    # between the two creases, so the grips are the bottom corners of THAT
    # strip, not the original hem corners. ----
    strip_left, strip_right = crease1_x, crease2_x
    crease3_y = bottom - (bottom - top) * hem_fraction
    bl = np.array([strip_left, bottom])
    br = np.array([strip_right, bottom])
    grips3 = [
        (bl, mirror_y(bl, crease3_y)),
        (br, mirror_y(br, crease3_y)),
    ]
    emit("fold hem up", grips3,
         ((strip_left, crease3_y), (strip_right, crease3_y)))

    return actions


# ---------------------------------------------------------------------------
# 4. VISUALISATION
# ---------------------------------------------------------------------------
COLORS = {
    "collar": (0, 255, 255),
    "left_sleeve": (255, 120, 0),
    "right_sleeve": (255, 120, 0),
    "hem_left": (200, 0, 255),
    "hem_right": (200, 0, 255),
    "hem_mid": (255, 0, 160),
}


def draw_analysis(img, mask, kp, actions, max_display=1100):
    vis = img.copy()

    # Tint the detected shirt region so segmentation errors are obvious.
    overlay = vis.copy()
    overlay[mask > 0] = (0, 200, 0)
    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)

    scale = min(1.0, max_display / float(max(vis.shape[:2])))
    r = max(4, int(9 / scale))
    th = max(2, int(3 / scale))
    fs = 0.7 / scale

    for name, pt in kp["image"].items():
        cv2.circle(vis, pt, r, COLORS.get(name, (255, 255, 255)), -1)
        cv2.putText(vis, name, (pt[0] + r + 4, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6,
                    COLORS.get(name, (255, 255, 255)), th)

    for i, act in enumerate(actions, 1):
        # The crease: the line the fabric actually folds along.
        c0 = _to_image(np.array(act["crease"][0]), kp["geom"]["M_inv"])
        c1 = _to_image(np.array(act["crease"][1]), kp["geom"]["M_inv"])
        cv2.line(vis, c0, c1, (255, 255, 255), th + 1, cv2.LINE_AA)
        cv2.putText(vis, f"crease {i}", c0, cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.55, (255, 255, 255), th)

        for pick, place in act["grips"]:
            cv2.arrowedLine(vis, pick, place, (0, 165, 255), th + 1,
                            tipLength=0.08)
            cv2.circle(vis, pick, r, (0, 255, 0), -1)
            cv2.circle(vis, place, r, (0, 0, 255), -1)

        first = act["grips"][0]
        mid = ((first[0][0] + first[1][0]) // 2,
               (first[0][1] + first[1][1]) // 2)
        cv2.putText(vis, f"{i}. {act['label']}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (0, 165, 255), th)

    if scale < 1.0:
        vis = cv2.resize(vis, (int(vis.shape[1] * scale),
                               int(vis.shape[0] * scale)))
    return vis


# ---------------------------------------------------------------------------
def analyze(img, dark_shirt=None, side_fraction=1.0/3.0,
            hem_fraction=0.5):
    """Full pipeline. Returns (mask, keypoints, actions) or raises ValueError."""
    mask = segment_shirt(img, dark_shirt=dark_shirt)
    if mask is None:
        raise ValueError(
            "No shirt found. The background probably doesn't contrast enough "
            "with the shirt. Try a plain dark mat under a light shirt (or "
            "vice versa), and re-run with --dark-shirt if the shirt is dark.")
    kp = find_keypoints(mask)
    if kp is None:
        raise ValueError("Found a region but couldn't read shirt landmarks "
                         "from it. Check the mask - it may not be a shirt.")
    return mask, kp, plan_folds(kp, side_fraction, hem_fraction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="analysis.png")
    ap.add_argument("--dark-shirt", action="store_true",
                    help="Force dark-shirt-on-light-background polarity")
    ap.add_argument("--save-mask", default=None,
                    help="Also write the binary mask, for debugging")
    ap.add_argument("--side-fraction", type=float, default=1.0/3.0,
                    help="How far in the side creases sit, as a fraction of "
                         "torso width. 0.333 = fold in thirds (the default "
                         "and what the tutorial does).")
    ap.add_argument("--hem-fraction", type=float, default=0.5,
                    help="Where the horizontal crease sits. 0.5 folds the "
                         "strip in half; 0.33 folds it in thirds.")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    dark = True if args.dark_shirt else None
    mask, kp, actions = analyze(img, dark_shirt=dark,
                                side_fraction=args.side_fraction,
                                hem_fraction=args.hem_fraction)

    print(f"Image: {img.shape[1]} x {img.shape[0]}")
    print(f"Shirt covers {100.0 * np.count_nonzero(mask) / mask.size:.1f}% "
          f"of the frame\n")
    print("KEYPOINTS (image pixels)")
    for name, pt in kp["image"].items():
        print(f"  {name:13s} {pt}")
    print("\nFOLD ACTIONS  (each fold = 2 grips, both moved together)")
    for i, act in enumerate(actions, 1):
        print(f"  {i}. {act['label']}")
        for j, (pick, place) in enumerate(act["grips"], 1):
            print(f"       grip {j}: pick={pick}  ->  place={place}")

    cv2.imwrite(args.out, draw_analysis(img, mask, kp, actions))
    print(f"\nAnnotated image written to {args.out}")
    print("LOOK AT IT before trusting the numbers - if the green outline "
          "isn't hugging the shirt, fix the lighting/background first.")

    if args.save_mask:
        cv2.imwrite(args.save_mask, mask)
        print(f"Mask written to {args.save_mask}")


if __name__ == "__main__":
    main()
