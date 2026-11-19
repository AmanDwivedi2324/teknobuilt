"""Step 2: quality gate. Decide, before any AI call, whether a page is worth processing."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Thresholds. Calibrated on the sample pages plus fake damaged copies (see README).
MIN_INK = 0.003         # less dark ink than this: blank page
MAX_INK = 0.70          # more than this: almost entirely dark (mud, shadow, bad scan)
DIRTY_INK = 0.15        # above this, keep going but flag the page
MIN_GRID_LINES = 20     # a real form has a table grid made of long straight lines
MIN_SHARPNESS = 100.0   # below this the scan is blurry
MAX_SKEW_DEG = 15.0     # Step 3 straightens small tilts; bigger ones get flagged
FAIL_PHRASES = ("FAILED TEST", "VOID", "DO NOT USE")


def measure(gray: np.ndarray) -> dict:
    """Cheap image statistics. No AI involved."""
    h, w = gray.shape
    ink_ratio = float((gray < 128).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Find long straight lines (the form's table grid). Works at any rotation.
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    lines = cv2.HoughLinesP(
        ink, 1, np.pi / 360, threshold=200, minLineLength=w // 4, maxLineGap=5
    )
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


def check_page(image_path: str | None, text_layer: str = "") -> dict:
    """Return {status: ok | warn | rejected, reasons: [...], metrics: {...}}."""
    reasons: list[str] = []
    warnings: list[str] = []

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if image_path else None
    if gray is None or min(gray.shape) < 500:
        return {"status": "rejected", "reasons": ["image missing, corrupted or too small"], "metrics": {}}

    m = measure(gray)
    text = text_layer.upper()

    if any(p in text for p in FAIL_PHRASES):
        reasons.append("page is marked as a failed or void form")
    if m["ink_ratio"] < MIN_INK:
        reasons.append("page is blank")
    elif m["ink_ratio"] > MAX_INK:
        reasons.append("page is almost entirely dark")
    elif m["grid_lines"] < MIN_GRID_LINES:
        reasons.append("no form grid found (not a safety form, or too damaged)")

    if m["ink_ratio"] > DIRTY_INK and not reasons:
        warnings.append("page is dirty; some fields may be unreadable")
    if m["sharpness"] < MIN_SHARPNESS:
        warnings.append("scan is blurry")
    if abs(m["skew_deg"]) > MAX_SKEW_DEG:
        warnings.append(f"page is tilted {m['skew_deg']} degrees")

    if reasons:
        status = "rejected"
    elif warnings:
        status = "warn"
    else:
        status = "ok"
    return {"status": status, "reasons": reasons + warnings, "metrics": m}


if __name__ == "__main__":
    from load_pdf import render_pages

    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "data/Sample_input.pdf")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "output/pages")
    for page in render_pages(pdf, out_dir):
        result = check_page(page["image"], page["text_layer"])
        print(f"page {page['page']}: {json.dumps(result)}")