# Tests for the quality gate

import numpy as np
import pytest
from fakes import rotate_cropped

from form_digitizer.models import Severity
from form_digitizer.quality import check_quality


def codes(report):
    return {i.code for i in report.issues}


@pytest.mark.parametrize("number", [1, 2, 3])
def test_real_forms_pass(sample_pages, number):
    report = check_quality(sample_pages[number].image, sample_pages[number].text_layer)
    assert not report.rejected
    assert report.metrics["grid_lines"] > 50


def test_failed_test_page_is_rejected_for_two_independent_reasons(sample_pages):
    page = sample_pages[4]
    report = check_quality(page.image, page.text_layer)
    assert report.rejected
    assert {"marked_void", "no_form_grid"} <= codes(report)


@pytest.mark.parametrize(
    ("image", "code"),
    [
        (None, "unreadable_image"),
        (np.full((100, 100), 255, np.uint8), "unreadable_image"),
        (np.full((2339, 1654), 245, np.uint8), "blank_page"),
        (np.full((2339, 1654), 30, np.uint8), "too_dark"),
    ],
    ids=["missing", "tiny", "blank", "black"],
)
def test_garbage_is_rejected(image, code):
    report = check_quality(image)
    assert report.rejected
    assert code in codes(report)


def test_heavy_tilt_is_noted_not_rejected(page1):
    report = check_quality(rotate_cropped(page1, 20))
    assert not report.rejected
    tilt = [i for i in report.issues if i.code == "tilted"]
    assert tilt and tilt[0].severity is Severity.INFO
    assert abs(abs(report.skew_deg) - 20) < 0.5
