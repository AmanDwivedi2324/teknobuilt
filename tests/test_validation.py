# Tests for validation and cross-check

from datetime import date

import pytest
from test_extraction import GOOD_REPLY

from form_digitizer.checkboxes import ChoiceReading
from form_digitizer.extraction.base import ReadResult
from form_digitizer.extraction.schema import FormRead, TextField, empty_read
from form_digitizer.layout import DEFAULT_LAYOUT
from form_digitizer.validation import build_form, clean_number, merge_choice, parse_date, text_value

TODAY = date(2026, 11, 19)
SHIFTS = DEFAULT_LAYOUT.shift_boxes


@pytest.mark.parametrize(
    ("raw", "iso", "code"),
    [
        ("2024-10-02", "2024-10-02", None),
        ("25/10/2024", "2024-10-25", None),  # day-first only
        ("02/10/2024", None, "ambiguous_date"),  # 2 Oct or 10 Feb?
        ("2031-01-01", "2031-01-01", "implausible_date"),
        ("next tuesday", None, "invalid_date"),
    ],
)
def test_dates(raw, iso, code):
    value, issue = parse_date(raw, TODAY)
    assert value == iso
    assert (issue.code if issue else None) == code


def test_numbers_are_cleaned_or_flagged():
    issues = []
    assert clean_number("employee_number", "78 52", issues) == "7852"
    assert clean_number("employee_number", "78S2", issues) == "78S2"
    assert [i.code for i in issues] == ["invalid_number"]


def test_illegible_and_missing_text_is_flagged_not_guessed():
    issues = []
    assert text_value("observer", TextField(value=None, status="illegible"), issues) is None
    assert text_value("observer", TextField(value=None, status="blank"), issues) is None
    assert text_value("comments", TextField(value=None, status="blank"), issues) is None  # optional field
    assert [i.code for i in issues] == ["illegible_field", "missing_field"]


def test_agreeing_readings_need_no_review():
    assert merge_choice("shift", ChoiceReading("Day"), "Day", True, SHIFTS) == ("Day", [])


def test_disagreement_is_flagged_and_pixels_win():
    value, issues = merge_choice("shift", ChoiceReading("Day"), "Night", True, SHIFTS)
    assert value == "Day" and [i.code for i in issues] == ["checkbox_mismatch"]


def test_unsure_pixels_defer_to_a_clear_model_answer():
    value, issues = merge_choice("shift", ChoiceReading(None, ["Day: faint"]), "Day", True, SHIFTS)
    assert value == "Day" and "checkbox_unclear" in {i.code for i in issues}


def test_value_not_on_the_form_is_rejected():
    value, issues = merge_choice("location_cwa", None, "9000", True, DEFAULT_LAYOUT.location_boxes)
    assert value is None and "invalid_value" in {i.code for i in issues}


def test_without_a_model_pixels_are_used_and_the_page_is_flagged_once():
    form, issues = build_form(None, ReadResult("mock", None, empty_read(), "no key"), DEFAULT_LAYOUT, TODAY)
    assert [i.code for i in issues] == ["text_not_read"]
    assert form.observer is None


def test_full_model_reply_builds_a_clean_record():
    read = ReadResult("model", "m", FormRead.model_validate(GOOD_REPLY))
    form, issues = build_form(None, read, DEFAULT_LAYOUT, TODAY)
    assert issues == []
    assert form.observation_date == "2024-10-02" and form.employee_number == "7852"
    assert [c.rating for c in form.checklist] == list("IIIRSIII")
    assert form.checklist[0].rating_label == "Imminent Hazard"
