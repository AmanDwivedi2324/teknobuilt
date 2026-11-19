import json

from conftest import SAMPLE_PDF, TRUTH
from test_extraction import GOOD_REPLY

from form_digitizer.extraction import build_reader
from form_digitizer.extraction.base import ReadResult
from form_digitizer.extraction.schema import FormRead
from form_digitizer.models import RunResult
from form_digitizer.pipeline import Pipeline


class FixedReader:
    """Always returns the correct reading of sample page 1."""

    def read(self, png):
        return ReadResult("model", "fixed", FormRead.model_validate(GOOD_REPLY))


class ExplodingReader:
    def read(self, png):
        raise RuntimeError("unexpected bug")


def test_end_to_end_without_a_key(settings):
    result = Pipeline(settings, build_reader(settings)).run(SAMPLE_PDF)

    assert [p.status for p in result.pages] == ["needs_review", "needs_review", "needs_review", "rejected"]
    for page in result.pages[:3]:
        assert {i.code for i in page.issues} == {"text_not_read"}  # the only reason is the missing key
        truth = TRUTH[page.page]
        assert page.form.location_cwa == truth["location"]
        assert [c.rating for c in page.form.checklist] == truth["checklist"]
    assert result.pages[3].form is None  # nothing extracted from a rejected page

    RunResult.model_validate(json.loads(result.model_dump_json()))  # output round-trips through its schema


def test_page_with_agreeing_readings_is_ok(settings, sample_pages):
    page = Pipeline(settings, FixedReader()).process_page(sample_pages[1])
    assert page.status == "ok" and page.issues == []
    assert page.form.observer == "John Davis" and page.form.shift == "Day"


def test_model_disagreeing_with_pixels_sends_page_to_review(settings, sample_pages):
    page = Pipeline(settings, FixedReader()).process_page(sample_pages[2])  # page 2 data != page 1 reply
    assert page.status == "needs_review"
    assert "checkbox_mismatch" in {i.code for i in page.issues}


def test_a_crash_on_one_page_does_not_stop_the_run(settings):
    result = Pipeline(settings, ExplodingReader()).run(SAMPLE_PDF)
    assert len(result.pages) == 4
    assert {i.code for p in result.pages[:3] for i in p.issues} == {"processing_error"}
    assert result.pages[3].status == "rejected"  # rejected before the reader is ever called


def test_unreadable_file_is_reported_not_raised(settings, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    result = Pipeline(settings, FixedReader()).run(bad)
    assert result.pages == [] and result.errors
