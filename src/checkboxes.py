"""Step 4: read every checkbox by counting ink pixels inside it. No AI involved."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from align import grid_masks

# ---- Form layout in the standard frame (measured once on aligned page 1) ----
# A box is (x0, y0, x1, y1) in pixels.
LOCATION_BOXES = {
    "1000": (515, 214, 642, 282), "2000": (768, 214, 895, 282), "3000": (1021, 214, 1147, 282),
    "4000": (1274, 214, 1400, 282), "8000": (1526, 214, 1649, 282),
    "5000": (515, 282, 642, 352), "5900": (768, 282, 895, 352), "6000": (1021, 282, 1147, 352),
    "7000": (1274, 282, 1400, 352), "OSLD": (1526, 282, 1649, 352),
}
SHIFT_BOXES = {"Day": (642, 352, 768, 426), "Night": (1147, 352, 1274, 426)}

CHECKLIST_COLUMNS = {"S": (1151, 1274), "R": (1274, 1397), "I": (1397, 1520), "N": (1520, 1640)}
CHECKLIST_TOP = 656          # top line of row 1
ROW_PITCH = 77.9             # row height
MAX_ROWS = 15                # the blank template has 15 rows; real forms use fewer

EXPECTATION_QUESTIONS = [
    "Has positive feedback provided?",
    "Has the job been stopped in the event of an imminent hazard?",
    "Has an investigation been conducted?",
    "Have corrective actions been taken?",
]
EXPECTATION_COLUMNS = {"Yes": (1270, 1393), "No": (1393, 1516), "N/A": (1516, 1638)}
EXPECTATION_ROWS_Y = [(1875, 1950), (1950, 2030), (2030, 2107), (2107, 2187)]

# ---- Thresholds (calibrated on the sample pages; see README) ----
MARGIN_PX = 8           # ignore the box border itself
EMPTY_MAX = 0.005       # less ink than this: empty box
MARK_MIN = 0.012        # more than this: a clear tick or X (real X marks measure 0.025-0.031)
SCRIBBLE_MIN = 0.30     # more than this: box filled in / scribbled over (crossed out)
DIRTY_BACKGROUND = 170  # paper inside the box darker than this: mud, can't trust it


def box_reading(gray: np.ndarray, ink: np.ndarray, seen: np.ndarray | None, box: tuple) -> dict:
    """How much ink is inside one box, and what that most likely means."""
    x0, y0, x1, y1 = box
    inner = (slice(y0 + MARGIN_PX, y1 - MARGIN_PX), slice(x0 + MARGIN_PX, x1 - MARGIN_PX))
    fill = float(ink[inner].mean() / 255)
    background = float(np.percentile(gray[inner], 90))  # the paper colour inside the box

    if seen is not None and seen[inner].min() == 0:
        state = "not scanned"      # part of the box is outside the scanned area
    elif background < DIRTY_BACKGROUND:
        state = "unreadable"
    elif fill >= SCRIBBLE_MIN:
        state = "scribbled"
    elif fill >= MARK_MIN:
        state = "marked"
    elif fill <= EMPTY_MAX:
        state = "empty"
    else:
        state = "faint"
    return {"state": state, "fill": round(fill, 3)}


def pick_one(readings: dict[str, dict]) -> dict:
    """Turn a group of boxes where only one should be marked into a single answer."""
    marked = [k for k, r in readings.items() if r["state"] == "marked"]
    flags = sorted({f"{k}: {r['state']}" for k, r in readings.items()
                    if r["state"] not in ("marked", "empty")})
    if len(marked) == 1:
        value = marked[0]
    elif len(marked) > 1:
        value = None
        flags.append(f"more than one box marked: {marked}")
    else:
        value = None
    return {
        "value": value,
        "needs_review": bool(flags),
        "flags": flags,
        "fill": {k: r["fill"] for k, r in readings.items()},
    }


def checklist_rows(vert: np.ndarray) -> list[tuple[int, int]]:
    """Find which of the 15 possible checklist rows exist on this form.

    A row exists if the vertical lines of its S/R/I/N boxes are drawn.
    """
    dividers = [x0 for x0, _ in CHECKLIST_COLUMNS.values()] + [CHECKLIST_COLUMNS["N"][1]]
    present = []
    for k in range(MAX_ROWS):
        y0 = round(CHECKLIST_TOP + k * ROW_PITCH)
        y1 = round(CHECKLIST_TOP + (k + 1) * ROW_PITCH)
        band = vert[y0 + 10: y1 - 10]
        lines = sum(band[:, max(x - 6, 0): x + 6].max(axis=1).mean() > 0.5 * 255 for x in dividers)
        present.append(lines >= 3)
    last = max((k for k, p in enumerate(present) if p), default=-1)
    return [
        (round(CHECKLIST_TOP + k * ROW_PITCH), round(CHECKLIST_TOP + (k + 1) * ROW_PITCH))
        for k in range(last + 1)
    ]


def read_checkboxes(aligned: np.ndarray, seen: np.ndarray | None = None) -> dict:
    """Read all checkbox groups on an aligned page."""
    ink = cv2.adaptiveThreshold(
        aligned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    horiz, vert = grid_masks(aligned)
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(horiz | vert))  # drop table lines, keep marks

    def group(boxes: dict) -> dict:
        return pick_one({k: box_reading(aligned, ink, seen, b) for k, b in boxes.items()})

    rows = []
    for i, (y0, y1) in enumerate(checklist_rows(vert), start=1):
        boxes = {k: (x0, y0, x1, y1) for k, (x0, x1) in CHECKLIST_COLUMNS.items()}
        rows.append({"row": i, **group(boxes)})

    expectations = []
    for question, (y0, y1) in zip(EXPECTATION_QUESTIONS, EXPECTATION_ROWS_Y):
        boxes = {k: (x0, y0, x1, y1) for k, (x0, x1) in EXPECTATION_COLUMNS.items()}
        expectations.append({"question": question, **group(boxes)})

    return {
        "location": group(LOCATION_BOXES),
        "shift": group(SHIFT_BOXES),
        "checklist": rows,
        "expectations": expectations,
    }


if __name__ == "__main__":
    from align import align_page
    from quality_gate import check_page

    for path in sys.argv[1:] or sorted(str(p) for p in Path("output/pages").glob("*.png")):
        name = Path(path).name
        gate = check_page(path)
        if gate["status"] == "rejected":
            print(f"== {name}: skipped ({gate['reasons']})")
            continue
        aligned, seen, info = align_page(cv2.imread(path, cv2.IMREAD_GRAYSCALE), gate["metrics"]["skew_deg"])
        if aligned is None:
            print(f"== {name}: could not align ({info['alignment_score']}), checkboxes not read")
            continue
        r = read_checkboxes(aligned, seen)
        groups = [r["location"], r["shift"], *r["checklist"], *r["expectations"]]
        print(f"== {name}")
        print("  location:", r["location"]["value"], "| shift:", r["shift"]["value"])
        print("  checklist:", [x["value"] for x in r["checklist"]])
        print("  expectations:", [x["value"] for x in r["expectations"]])
        print("  needs review:", [g["flags"] for g in groups if g["needs_review"]] or "none")