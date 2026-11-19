"""Step 3: straighten each page and map it onto one standard frame."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Standard frame: every aligned page has this size, so a box is always at the same spot.
CANVAS_W, CANVAS_H = 1654, 2339

# Table lines every form of this type has, in the standard frame (measured once on a
# clean page). Used to fine-tune the alignment and to check that it worked.
EXPECTED_LINES_Y = [1, 72, 143, 214, 282, 352, 426, 501, 579, 656, 1839, 1875, 1950, 2030, 2107, 2187, 2336]
EXPECTED_LINES_X = [4, 82, 390, 642, 768, 895, 1149, 1273, 1397, 1518, 1639]
SNAP_TOLERANCE_PX = 25      # how far off a line may be before fine-tuning
LINE_TOLERANCE_PX = 12      # how far off a line may be after fine-tuning
MIN_ALIGNMENT_SCORE = 0.6   # mud or blur can hide some lines; below this, don't trust positions


def rotate(img: np.ndarray, angle_deg: float, fill: int = 255) -> np.ndarray:
    """Rotate around the centre, growing the canvas so no corner is cut off."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=fill)


def grid_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only long horizontal and long vertical strokes: the form's table lines."""
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40)))
    return horiz, vert


def line_positions(mask: np.ndarray, axis: int, min_length: float) -> list[int]:
    """Centre of each line in a line mask. axis=1 gives y of horizontal lines, axis=0 gives x of vertical ones."""
    profile = mask.sum(axis=axis) / 255
    groups: list[list[int]] = []
    for i in np.where(profile > min_length)[0]:
        if groups and i - groups[-1][-1] <= 3:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [int(np.mean(g)) for g in groups]


def find_lines(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """(x of vertical lines, y of horizontal lines)."""
    horiz, vert = grid_masks(gray)
    return line_positions(vert, axis=0, min_length=150), line_positions(horiz, axis=1, min_length=0.3 * gray.shape[1])


def fit_axis(found: list[int], expected: list[int]) -> tuple[float, float]:
    """Match found lines to expected ones; fit expected = a * found + b."""
    pairs = []
    for e in expected:
        near = [f for f in found if abs(f - e) <= SNAP_TOLERANCE_PX]
        if near:
            pairs.append((min(near, key=lambda f: abs(f - e)), e))
    if len(pairs) < 3:
        return 1.0, 0.0  # not enough evidence: keep the first guess
    f, e = np.array(pairs, dtype=float).T
    a, b = np.polyfit(f, e, 1)
    return float(a), float(b)


def alignment_score(aligned: np.ndarray) -> float:
    """Share of the expected lines (both directions) found where they should be."""
    found_x, found_y = find_lines(aligned)
    hits = sum(any(abs(f - e) <= LINE_TOLERANCE_PX for f in found_y) for e in EXPECTED_LINES_Y)
    hits += sum(any(abs(f - e) <= LINE_TOLERANCE_PX for f in found_x) for e in EXPECTED_LINES_X)
    return hits / (len(EXPECTED_LINES_Y) + len(EXPECTED_LINES_X))


def warp_to_frame(gray: np.ndarray, angle_deg: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rotate, stretch the form's grid onto the frame, then fine-tune using known lines.

    Also returns a 'seen' mask: 255 where the pixel came from the scan, 0 where the
    form was outside the scanned area (e.g. a corner cut off by a tilted scan).
    """
    straight = rotate(gray, angle_deg)
    seen = rotate(np.full_like(gray, 255), angle_deg, fill=0)
    horiz, vert = grid_masks(straight)
    ys, xs = np.nonzero(horiz | vert)
    if len(xs) == 0:
        return None, None

    # First guess: stretch the grid's bounding box onto the frame.
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sx, sy = CANVAS_W / (x1 - x0), CANVAS_H / (y1 - y0)
    m = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
    first = cv2.warpAffine(straight, m[:2], (CANVAS_W, CANVAS_H), borderValue=255)

    # Fine-tune: if a border was cut off, the box is wrong; known inner lines fix it.
    found_x, found_y = find_lines(first)
    ax, bx = fit_axis(found_x, EXPECTED_LINES_X)
    ay, by = fit_axis(found_y, EXPECTED_LINES_Y)
    m = np.array([[ax, 0, bx], [0, ay, by], [0, 0, 1]]) @ m

    aligned = cv2.warpAffine(straight, m[:2], (CANVAS_W, CANVAS_H), borderValue=255)
    seen = cv2.warpAffine(seen, m[:2], (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST, borderValue=0)
    return aligned, seen


def align_page(gray: np.ndarray, skew_deg: float) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Try the measured skew plus quarter turns (sideways / upside-down scans); keep the best."""
    best, best_seen, best_score, best_angle = None, None, -1.0, 0.0
    for turn in (0, 90, 180, 270):
        angle = skew_deg + turn
        aligned, seen = warp_to_frame(gray, angle)
        if aligned is None:
            continue
        score = alignment_score(aligned)
        if score > best_score:
            best, best_seen, best_score, best_angle = aligned, seen, score, angle
        if score == 1.0:
            break  # perfect match, no need to try other turns

    ok = best_score >= MIN_ALIGNMENT_SCORE
    info = {
        "aligned": ok,
        "alignment_score": round(max(best_score, 0.0), 2),
        "rotation_applied_deg": round(best_angle, 2),
        "scanned_fraction": round(float((best_seen > 0).mean()), 3) if best_seen is not None else 0.0,
    }
    return (best if ok else None), (best_seen if ok else None), info


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
        aligned, seen, info = align_page(gray, gate["metrics"]["skew_deg"])
        if aligned is not None:
            cv2.imwrite(str(out_dir / Path(path).name), aligned)
        print(Path(path).name, json.dumps(info))