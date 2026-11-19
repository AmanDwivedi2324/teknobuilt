# Step 3: straighten page, map to standard frame

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import DEFAULT_LAYOUT, FormLayout

SNAP_TOLERANCE_PX = 25  # allowed offset before fine-tune
LINE_TOLERANCE_PX = 12  # allowed offset after fine-tune
MIN_ALIGNMENT_SCORE = 0.6  # below this, don't trust positions


@dataclass
class AlignmentResult:
    aligned: np.ndarray | None  # None if alignment failed
    seen: np.ndarray | None  # 0 = outside the scan
    score: float
    rotation_deg: float
    scanned_fraction: float

    @property
    def ok(self) -> bool:
        return self.aligned is not None

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "aligned": self.ok,
            "alignment_score": round(self.score, 2),
            "rotation_applied_deg": round(self.rotation_deg, 2),
            "scanned_fraction": round(self.scanned_fraction, 3),
        }


# rotate without cutting corners
def rotate(img: np.ndarray, angle_deg: float, fill: int = 255) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=fill)


# keep only the table lines
def grid_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40)))
    return horiz, vert


# centre position of each line
def line_positions(mask: np.ndarray, axis: int, min_length: float) -> list[int]:
    profile = mask.sum(axis=axis) / 255
    groups: list[list[int]] = []
    for i in np.where(profile > min_length)[0]:
        if groups and i - groups[-1][-1] <= 3:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [int(np.mean(g)) for g in groups]


# x of vertical, y of horizontal lines
def find_lines(gray: np.ndarray) -> tuple[list[int], list[int]]:
    horiz, vert = grid_masks(gray)
    return (
        line_positions(vert, axis=0, min_length=150),
        line_positions(horiz, axis=1, min_length=0.3 * gray.shape[1]),
    )


# fit found lines onto expected ones
def fit_axis(found: list[int], expected: tuple[int, ...]) -> tuple[float, float]:
    pairs = []
    for e in expected:
        near = [f for f in found if abs(f - e) <= SNAP_TOLERANCE_PX]
        if near:
            pairs.append((min(near, key=lambda f: abs(f - e)), e))
    if len(pairs) < 3:
        return 1.0, 0.0  # too few matches, keep guess
    f, e = np.array(pairs, dtype=float).T
    a, b = np.polyfit(f, e, 1)
    return float(a), float(b)


# share of expected lines in place
def alignment_score(aligned: np.ndarray, layout: FormLayout) -> float:
    found_x, found_y = find_lines(aligned)
    hits = sum(any(abs(f - e) <= LINE_TOLERANCE_PX for f in found_y) for e in layout.expected_lines_y)
    hits += sum(any(abs(f - e) <= LINE_TOLERANCE_PX for f in found_x) for e in layout.expected_lines_x)
    return hits / (len(layout.expected_lines_y) + len(layout.expected_lines_x))


# rotate, stretch, then fine-tune
def warp_to_frame(
    gray: np.ndarray, angle_deg: float, layout: FormLayout
) -> tuple[np.ndarray | None, np.ndarray | None]:
    width, height = layout.canvas_size
    straight = rotate(gray, angle_deg)
    seen = rotate(np.full_like(gray, 255), angle_deg, fill=0)
    horiz, vert = grid_masks(straight)
    ys, xs = np.nonzero(horiz | vert)
    if len(xs) == 0:
        return None, None

    # first guess from the grid's box
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sx, sy = width / (x1 - x0), height / (y1 - y0)
    m = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
    first = cv2.warpAffine(straight, m[:2], (width, height), borderValue=255)

    # fine-tune using known inner lines
    found_x, found_y = find_lines(first)
    ax, bx = fit_axis(found_x, layout.expected_lines_x)
    ay, by = fit_axis(found_y, layout.expected_lines_y)
    m = np.array([[ax, 0, bx], [0, ay, by], [0, 0, 1]]) @ m

    aligned = cv2.warpAffine(straight, m[:2], (width, height), borderValue=255)
    seen = cv2.warpAffine(seen, m[:2], (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    return aligned, seen


# try skew plus quarter turns, keep best
def align_page(gray: np.ndarray, skew_deg: float, layout: FormLayout = DEFAULT_LAYOUT) -> AlignmentResult:
    best, best_seen, best_score, best_angle = None, None, 0.0, 0.0
    for turn in (0, 90, 180, 270):
        angle = skew_deg + turn
        aligned, seen = warp_to_frame(gray, angle, layout)
        if aligned is None:
            continue
        score = alignment_score(aligned, layout)
        if score > best_score:
            best, best_seen, best_score, best_angle = aligned, seen, score, angle
        if score == 1.0:
            break  # perfect match, stop here

    scanned = float((best_seen > 0).mean()) if best_seen is not None else 0.0
    if best_score < MIN_ALIGNMENT_SCORE:
        return AlignmentResult(None, None, best_score, best_angle, scanned)
    return AlignmentResult(best, best_seen, best_score, best_angle, scanned)
