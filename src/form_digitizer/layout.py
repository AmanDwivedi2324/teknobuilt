"""Where everything sits on the form, in the standard frame produced by alignment.

All positions were measured once on a clean, aligned sample page. Supporting a new form
version means adding another FormLayout here; no pipeline code has to change.
"""

from dataclasses import dataclass

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
Span = tuple[int, int]  # start, end in pixels


@dataclass(frozen=True)
class FormLayout:
    name: str
    canvas_size: tuple[int, int]  # width, height of the standard frame

    # Table lines every form of this type has; used to fine-tune and verify alignment.
    expected_lines_x: tuple[int, ...]
    expected_lines_y: tuple[int, ...]

    location_boxes: dict[str, Box]
    shift_boxes: dict[str, Box]

    checklist_columns: dict[str, Span]  # rating code -> x span
    checklist_top: int  # y of the top line of row 1
    row_pitch: float  # row height
    max_rows: int

    expectation_questions: tuple[str, ...]
    expectation_columns: dict[str, Span]  # answer -> x span
    expectation_rows_y: tuple[Span, ...]


RATING_LABELS = {"S": "Safe", "R": "At Risk", "I": "Imminent Hazard", "N": "N/A"}

SAFETY_OBSERVATION_V1 = FormLayout(
    name="pace-safety-observation-v1",
    canvas_size=(1654, 2339),  # A4 at 200 DPI
    expected_lines_x=(4, 82, 390, 642, 768, 895, 1149, 1273, 1397, 1518, 1639),
    expected_lines_y=(
        1,
        72,
        143,
        214,
        282,
        352,
        426,
        501,
        579,
        656,
        1839,
        1875,
        1950,
        2030,
        2107,
        2187,
        2336,
    ),
    location_boxes={
        "1000": (515, 214, 642, 282),
        "2000": (768, 214, 895, 282),
        "3000": (1021, 214, 1147, 282),
        "4000": (1274, 214, 1400, 282),
        "8000": (1526, 214, 1649, 282),
        "5000": (515, 282, 642, 352),
        "5900": (768, 282, 895, 352),
        "6000": (1021, 282, 1147, 352),
        "7000": (1274, 282, 1400, 352),
        "OSLD": (1526, 282, 1649, 352),
    },
    shift_boxes={"Day": (642, 352, 768, 426), "Night": (1147, 352, 1274, 426)},
    checklist_columns={"S": (1151, 1274), "R": (1274, 1397), "I": (1397, 1520), "N": (1520, 1640)},
    checklist_top=656,
    row_pitch=77.9,
    max_rows=15,  # the blank template has 15 rows; filled forms use fewer
    expectation_questions=(
        "Has positive feedback provided?",
        "Has the job been stopped in the event of an imminent hazard?",
        "Has an investigation been conducted?",
        "Have corrective actions been taken?",
    ),
    expectation_columns={"Yes": (1270, 1393), "No": (1393, 1516), "N/A": (1516, 1638)},
    expectation_rows_y=((1875, 1950), (1950, 2030), (2030, 2107), (2107, 2187)),
)

DEFAULT_LAYOUT = SAFETY_OBSERVATION_V1
