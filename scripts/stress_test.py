"""Robustness report: damage the sample pages on purpose and see how the pipeline copes.

Covers the deterministic part (quality gate -> alignment -> checkbox reading), compared with
what is really marked on the forms. The headline column is "wrong, not flagged": it must be 0.

    python scripts/stress_test.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import SAMPLE_PDF, TRUTH  # noqa: E402
from fakes import draw_x, mud, perspective, rotate_cropped, scribble  # noqa: E402

from form_digitizer.alignment import align_page  # noqa: E402
from form_digitizer.checkboxes import read_checkboxes  # noqa: E402
from form_digitizer.layout import DEFAULT_LAYOUT as L  # noqa: E402
from form_digitizer.pdf_loader import load_pages  # noqa: E402
from form_digitizer.quality import check_quality  # noqa: E402


def cell(row: int, column: str) -> tuple[int, int, int, int]:
    x0, x1 = L.checklist_columns[column]
    y0 = round(L.checklist_top + (row - 1) * L.row_pitch)
    return x0, y0, x1, round(y0 + L.row_pitch)


def edited(page: np.ndarray) -> np.ndarray:
    """Page 1 with row 4's answer scribbled out and re-marked, and two marks in row 6."""
    aligned = align_page(page, 0).aligned.copy()
    scribble(aligned, cell(4, "R"))
    draw_x(aligned, cell(4, "S"))
    draw_x(aligned, cell(6, "N"))
    return aligned


def evaluate(name: str, image: np.ndarray, truth_page: int) -> str:
    gate = check_quality(image)
    if gate.rejected:
        return f"| {name} | rejected ({', '.join(i.code for i in gate.issues)}) | - | - | - | - |"
    result = align_page(image, gate.skew_deg)
    if not result.ok:
        return f"| {name} | passed | no (score {result.score:.2f}) -> flagged | - | - | 0 |"

    report = read_checkboxes(result.aligned, result.seen)
    t = TRUTH[truth_page]
    want = [t["location"], t["shift"], *t["checklist"], *t["expectations"]]
    groups = [report.location, report.shift, *report.checklist, *report.expectations]
    if len(groups) != len(want):
        return f"| {name} | passed | yes | row count {len(groups)} vs {len(want)} | - | ? |"
    correct = sum(g.value == w for g, w in zip(groups, want, strict=True))
    flagged = sum(g.needs_review for g in groups)
    silent = sum(g.value != w and not g.needs_review for g, w in zip(groups, want, strict=True))
    if name.startswith("Crossed-out"):
        silent = sum(  # rows 4 and 6 were changed on purpose; judge them by "flagged", not "correct"
            g.value != w and not g.needs_review and i not in (5, 7)
            for i, (g, w) in enumerate(zip(groups, want, strict=True))
        )
    warn = ", ".join(i.code for i in gate.issues) or "ok"
    return (
        f"| {name} | {warn} | yes ({result.score:.2f}) | {correct}/{len(want)} | {flagged} | **{silent}** |"
    )


def main() -> None:
    pages = {p.number: p.image for p in load_pages(SAMPLE_PDF)}
    p1, p2 = pages[1], pages[2]
    rng = np.random.default_rng(0)
    cases = [
        ("Clean page 1", p1, 1),
        ("Clean page 2", p2, 2),
        ("Clean page 3", pages[3], 3),
        ("Tilted 3°", rotate_cropped(p1, 3), 1),
        ("Tilted 8°, corner cut off", rotate_cropped(p1, -8), 1),
        ("Tilted 20°, side cut off", rotate_cropped(p1, 20), 1),
        ("Upside down", cv2.rotate(p2, cv2.ROTATE_180), 2),
        ("Sideways", cv2.rotate(p2, cv2.ROTATE_90_CLOCKWISE), 2),
        ("Blurred", cv2.GaussianBlur(p1, (0, 0), 4), 1),
        ("Light mud", mud(p1, 12, 0.6), 1),
        ("Medium mud", mud(p1, 30, 0.85), 1),
        ("Heavy mud", mud(p1, 45, 0.9), 1),
        ("Very heavy mud", mud(p1, 60, 0.95), 1),
        ("Phone photo at an angle", perspective(p2), 2),
        ("Crossed-out answer + double mark", edited(p1), 1),
        ("Blank page", np.clip(245 + rng.normal(0, 4, p1.shape), 0, 255).astype(np.uint8), 1),
        ("Black page", np.full(p1.shape, 30, np.uint8), 1),
        ("Corrupted image", None, 1),
    ]
    print("| Case | Quality gate | Aligned | Checkboxes correct | Flagged for review | Wrong, not flagged |")
    print("|---|---|---|---|---|---|")
    for name, image, truth_page in cases:
        print(evaluate(name, image, truth_page))


if __name__ == "__main__":
    main()
