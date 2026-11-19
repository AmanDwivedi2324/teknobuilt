"""Step 4: read every checkbox by counting ink pixels inside it. No AI involved."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .alignment import grid_masks
from .layout import DEFAULT_LAYOUT, Box, FormLayout

# Thresholds, calibrated on the sample pages (real X marks measure 0.025-0.031).
MARGIN_PX = 8  # ignore the box border itself
EMPTY_MAX = 0.005  # less ink than this: empty box
MARK_MIN = 0.012  # more than this: a clear tick or X
SCRIBBLE_MIN = 0.30  # more than this: box filled in / scribbled over (crossed out)
DIRTY_BACKGROUND = 170  # paper inside the box darker than this: mud, can't trust it


@dataclass
class ChoiceReading:
    """The answer for a group of boxes where exactly one should be marked."""

    value: str | None
    flags: list[str] = field(default_factory=list)  # anything a person should check
    fill: dict[str, float] = field(default_factory=dict)  # ink share per box, for debugging

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


@dataclass
class CheckboxReport:
    location: ChoiceReading
    shift: ChoiceReading
    checklist: list[ChoiceReading]  # index 0 = row 1
    expectations: list[ChoiceReading]  # same order as layout.expectation_questions


def box_state(gray: np.ndarray, ink: np.ndarray, seen: np.ndarray | None, box: Box) -> tuple[str, float]:
    """(state, ink share) for one box."""
    x0, y0, x1, y1 = box
    inner = (slice(y0 + MARGIN_PX, y1 - MARGIN_PX), slice(x0 + MARGIN_PX, x1 - MARGIN_PX))
    fill = float(ink[inner].mean() / 255)
    background = float(np.percentile(gray[inner], 90))  # the paper colour inside the box

    if seen is not None and seen[inner].min() == 0:
        state = "not scanned"  # part of the box is outside the scanned area
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
    return state, round(fill, 3)


def pick_one(states: dict[str, tuple[str, float]]) -> ChoiceReading:
    marked = [k for k, (state, _) in states.items() if state == "marked"]
    flags = sorted({f"{k}: {state}" for k, (state, _) in states.items() if state not in ("marked", "empty")})
    value = marked[0] if len(marked) == 1 else None
    if len(marked) > 1:
        flags.append(f"more than one box marked: {marked}")
    return ChoiceReading(value, flags, {k: fill for k, (_, fill) in states.items()})


def checklist_rows(vert: np.ndarray, layout: FormLayout) -> list[tuple[int, int]]:
    """Find which checklist rows exist: a row exists if its S/R/I/N box borders are drawn."""
    columns = list(layout.checklist_columns.values())
    dividers = [x0 for x0, _ in columns] + [columns[-1][1]]
    present = []
    for k in range(layout.max_rows):
        y0 = round(layout.checklist_top + k * layout.row_pitch)
        y1 = round(layout.checklist_top + (k + 1) * layout.row_pitch)
        band = vert[y0 + 10 : y1 - 10]
        lines = sum(band[:, max(x - 6, 0) : x + 6].max(axis=1).mean() > 0.5 * 255 for x in dividers)
        present.append(lines >= 3)
    last = max((k for k, p in enumerate(present) if p), default=-1)
    return [
        (
            round(layout.checklist_top + k * layout.row_pitch),
            round(layout.checklist_top + (k + 1) * layout.row_pitch),
        )
        for k in range(last + 1)
    ]


def read_checkboxes(
    aligned: np.ndarray, seen: np.ndarray | None = None, layout: FormLayout = DEFAULT_LAYOUT
) -> CheckboxReport:
    ink = cv2.adaptiveThreshold(aligned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    horiz, vert = grid_masks(aligned)
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(horiz | vert))  # drop table lines, keep marks

    def group(boxes: dict[str, Box]) -> ChoiceReading:
        return pick_one({k: box_state(aligned, ink, seen, b) for k, b in boxes.items()})

    def row_boxes(columns: dict[str, tuple[int, int]], y0: int, y1: int) -> dict[str, Box]:
        return {k: (x0, y0, x1, y1) for k, (x0, x1) in columns.items()}

    return CheckboxReport(
        location=group(layout.location_boxes),
        shift=group(layout.shift_boxes),
        checklist=[
            group(row_boxes(layout.checklist_columns, y0, y1)) for y0, y1 in checklist_rows(vert, layout)
        ],
        expectations=[
            group(row_boxes(layout.expectation_columns, y0, y1)) for y0, y1 in layout.expectation_rows_y
        ],
    )
