import cv2
import pytest
from fakes import mud, perspective, rotate_cropped

from form_digitizer.alignment import align_page
from form_digitizer.quality import check_quality


def align(image):
    return align_page(image, check_quality(image).skew_deg)


def test_clean_page_aligns_perfectly(aligned_pages):
    for result in aligned_pages.values():
        assert result.ok and result.score == 1.0 and result.scanned_fraction == 1.0


@pytest.mark.parametrize("degrees", [3, -8, 20])
def test_tilted_pages_are_straightened(page1, degrees):
    result = align(rotate_cropped(page1, degrees))
    assert result.ok and result.score >= 0.9
    assert result.scanned_fraction < 1.0  # corners fell outside the scan, and we know it


@pytest.mark.parametrize(
    ("rotation", "expected_deg"),
    [(cv2.ROTATE_180, 180), (cv2.ROTATE_90_CLOCKWISE, 90)],
    ids=["upside-down", "sideways"],
)
def test_upside_down_and_sideways_scans(page1, rotation, expected_deg):
    result = align(cv2.rotate(page1, rotation))
    assert result.ok and result.score == 1.0
    assert result.rotation_deg == pytest.approx(expected_deg, abs=0.5)


def test_moderate_mud_still_aligns(page1):
    assert align(mud(page1, blobs=30, strength=0.85)).ok


def test_unsupported_damage_is_refused_not_guessed(page1):
    """Perspective (phone photo) isn't corrected; alignment must say so instead of returning a bad frame."""
    result = align(perspective(page1))
    assert not result.ok and result.aligned is None
