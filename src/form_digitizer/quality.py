"""Step 2: quality gate. Decide, before any model call, whether a page is worth processing.

Rule of thumb: if the form's table grid is still visible, process the page and flag what is
wrong with it; if it is not, reject the page. Rejected pages cost nothing and can't be
"read" into invented data.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import Issue, Severity

# Thresholds, calibrated on the sample pages plus synthetic damaged copies (see README).
MIN_SIDE_PX = 500  # smaller than this is not a usable scan
MIN_INK = 0.003  # less dark ink than this: blank page
MAX_INK = 0.70  # more than this: almost entirely dark (mud, shadow, bad scan)
DIRTY_INK = 0.15  # above this, keep going but flag the page
MIN_GRID_LINES = 20  # a real form has a table grid made of long straight lines
MIN_SHARPNESS = 100.0  # below this the scan is blurry (rough signal: noise raises it)
MAX_SKEW_DEG = 15.0  # alignment straightens any tilt; bigger tilts are just noted
FAIL_PHRASES = ("FAILED TEST", "VOID", "DO NOT USE")


@dataclass
class QualityReport:
    issues: list[Issue]
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return any(i.severity is Severity.REJECT for i in self.issues)

    @property
    def skew_deg(self) -> float:
        return float(self.metrics.get("skew_deg", 0.0))


def measure(gray: np.ndarray) -> dict[str, float | int]:
    """Cheap image statistics. No AI involved."""
    h, w = gray.shape
    ink_ratio = float((gray < 128).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Long straight lines (the form's table grid). Works at any rotation.
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    lines = cv2.HoughLinesP(ink, 1, np.pi / 360, threshold=200, minLineLength=w // 4, maxLineGap=5)
    lines = [] if lines is None else lines.reshape(-1, 4)  # shape differs across OpenCV versions

    # Skew = typical angle of those lines, folded so vertical lines count too.
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines]
    folded = [((a + 45) % 90) - 45 for a in angles]
    skew = float(np.median(folded)) if folded else 0.0

    return {
        "ink_ratio": round(ink_ratio, 4),
        "sharpness": round(sharpness, 1),
        "grid_lines": len(lines),
        "skew_deg": round(skew, 2),
    }


def check_quality(gray: np.ndarray | None, text_layer: str = "") -> QualityReport:
    if gray is None or gray.ndim != 2 or min(gray.shape) < MIN_SIDE_PX:
        return QualityReport(
            [_issue("unreadable_image", "image missing, corrupted or too small", Severity.REJECT)]
        )

    m = measure(gray)
    issues: list[Issue] = []

    if any(p in text_layer.upper() for p in FAIL_PHRASES):
        issues.append(_issue("marked_void", "page is marked as a failed or void form", Severity.REJECT))
    if m["ink_ratio"] < MIN_INK:
        issues.append(_issue("blank_page", "page is blank", Severity.REJECT))
    elif m["ink_ratio"] > MAX_INK:
        issues.append(_issue("too_dark", "page is almost entirely dark", Severity.REJECT))
    elif m["grid_lines"] < MIN_GRID_LINES:
        issues.append(
            _issue("no_form_grid", "no form grid found (not a safety form, or too damaged)", Severity.REJECT)
        )

    rejected = any(i.severity is Severity.REJECT for i in issues)
    if m["ink_ratio"] > DIRTY_INK and not rejected:
        issues.append(_issue("dirty_page", "page is dirty; some fields may be unreadable", Severity.REVIEW))
    if m["sharpness"] < MIN_SHARPNESS:
        issues.append(_issue("blurry", "scan is blurry", Severity.REVIEW))
    if abs(m["skew_deg"]) > MAX_SKEW_DEG:
        issues.append(
            _issue("tilted", f"page was tilted {m['skew_deg']} degrees (straightened)", Severity.INFO)
        )

    return QualityReport(issues, m)


def _issue(code: str, message: str, severity: Severity) -> Issue:
    return Issue(code=code, message=message, severity=severity)
