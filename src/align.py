"""Step 3: straighten each page and map it onto one standard frame."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Standard frame: every aligned page has this size, so a box is always at the same spot.
CANVAS_W, CANVAS_H = 1654, 2339

# Horizontal lines every form of this type has, in the standard frame
# (measured once on a clean page). Used to check that alignment worked.
EXPECTED_LINES_Y = [1, 72, 143, 214, 282, 352, 426, 501, 579, 656, 1839, 1875, 1950, 2030, 2107, 2187, 2336]
LINE_TOLERANCE_PX = 12
MIN_ALIGNMENT_SCORE = 0.6   # mud or blur can hide some lines; below this, don't trust positions


def rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around the centre, growing the canvas so no corner is cut off."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(gray, m, (new_w, new_h), borderValue=255)


def grid_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only long horizontal and long vertical strokes: the form's table lines."""
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40)))
    return horiz, vert


def horizontal_line_ys(horiz: np.ndarray, min_fraction: float = 0.3) -> list[int]:
    """Y positions of lines that run across at least `min_fraction` of the page."""
    h, w = horiz.shape
    rows = np.where(horiz.sum(axis=1) / 255 > w * min_fraction)[0]
    groups: list[list[int]] = []
    for y in rows:
        if groups and y - groups[-1][-1] <= 3:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    return [int(np.mean(g)) for g in groups]


def alignment_score(aligned: np.ndarray) -> float:
    """Share of the expected lines found where they should be."""
    found = horizontal_line_ys(grid_masks(aligned)[0])
    hits = sum(any(abs(f - e) <= LINE_TOLERANCE_PX for f in found) for e in EXPECTED_LINES_Y)
    return hits / len(EXPECTED_LINES_Y)


def warp_to_frame(gray: np.ndarray, angle_deg: float) -> np.ndarray | None:
    """Rotate, crop to the outer edge of the form's grid, resize to the standard frame."""
    straight = rotate(gray, angle_deg)
    horiz, vert = grid_masks(straight)
    ys, xs = np.nonzero(horiz | vert)
    if len(xs) == 0:
        return None
    crop = straight[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return cv2.resize(crop, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)


def align_page(gray: np.ndarray, skew_deg: float) -> tuple[np.ndarray | None, dict]:
    """Try the measured skew plus quarter turns (sideways / upside-down scans); keep the best."""
    best, best_score, best_angle = None, -1.0, 0.0
    for turn in (0, 90, 180, 270):
        angle = skew_deg + turn
        aligned = warp_to_frame(gray, angle)
        if aligned is None:
            continue
        score = alignment_score(aligned)
        if score > best_score:
            best, best_score, best_angle = aligned, score, angle
        if score == 1.0:
            break  # perfect match, no need to try other turns

    info = {
        "aligned": best_score >= MIN_ALIGNMENT_SCORE,
        "alignment_score": round(max(best_score, 0.0), 2),
        "rotation_applied_deg": round(best_angle, 2),
    }
    return (best if info["aligned"] else None), info


if __name__ == "__main__":
    from quality_gate import check_page

    out_dir = Path("output/aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sys.argv[1:] or sorted(str(p) for p in Path("output/pages").glob("*.png")):
        gate = check_page(path)
        if gate["status"] == "rejected":
            print(Path(path).name, "skipped:", gate["reasons"])
            continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        aligned, info = align_page(gray, gate["metrics"]["skew_deg"])
        if aligned is not None:
            cv2.imwrite(str(out_dir / Path(path).name), aligned)
        print(Path(path).name, json.dumps(info))