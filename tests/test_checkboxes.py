# Tests for checkbox reading

import pytest
from conftest import TRUTH
from fakes import draw_x, rotate_cropped, scribble

from form_digitizer.alignment import align_page
from form_digitizer.checkboxes import read_checkboxes
from form_digitizer.layout import DEFAULT_LAYOUT as L
from form_digitizer.quality import check_quality


def values(report):
    return {
        "location": report.location.value,
        "shift": report.shift.value,
        "checklist": [r.value for r in report.checklist],
        "expectations": [r.value for r in report.expectations],
    }


def all_groups(report):
    return [report.location, report.shift, *report.checklist, *report.expectations]


@pytest.mark.parametrize("number", [1, 2, 3])
def test_sample_forms_read_exactly(aligned_pages, number):
    result = aligned_pages[number]
    report = read_checkboxes(result.aligned, result.seen)
    assert values(report) == TRUTH[number]
    assert not any(g.needs_review for g in all_groups(report))


@pytest.mark.parametrize("degrees", [-8, 20])
def test_cut_off_scans_never_give_silent_wrong_answers(page1, degrees):
    image = rotate_cropped(page1, degrees)
    aligned = align_page(image, check_quality(image).skew_deg)
    report = read_checkboxes(aligned.aligned, aligned.seen)
    truth = TRUTH[1]
    expected = [truth["location"], truth["shift"], *truth["checklist"], *truth["expectations"]]
    for group, want in zip(all_groups(report), expected, strict=True):
        assert group.value == want or group.needs_review


def cell(row: int, column: str):
    x0, x1 = L.checklist_columns[column]
    y0 = round(L.checklist_top + (row - 1) * L.row_pitch)
    return x0, y0, x1, round(y0 + L.row_pitch)


def test_crossed_out_answer_and_double_mark(aligned_pages):
    page = aligned_pages[1].aligned.copy()
    scribble(page, cell(4, "R"))  # cross out R
    draw_x(page, cell(4, "S"))  # mark S instead
    draw_x(page, cell(6, "N"))  # second mark in row 6

    report = read_checkboxes(page)
    row4, row6 = report.checklist[3], report.checklist[5]
    assert row4.value == "S" and "R: scribbled" in row4.flags
    assert row6.value is None and any("more than one" in f for f in row6.flags)
