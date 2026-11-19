# Shared test fixtures and truth

from pathlib import Path

import pytest

from form_digitizer.alignment import align_page
from form_digitizer.config import Settings
from form_digitizer.pdf_loader import load_pages
from form_digitizer.quality import check_quality

SAMPLE_PDF = Path(__file__).parents[1] / "data" / "Sample_input.pdf"

# what is really marked on the samples
TRUTH = {
    1: {
        "location": "8000",
        "shift": "Day",
        "checklist": list("IIIRSIII"),
        "expectations": ["Yes", "No", "N/A", "N/A"],
    },
    2: {
        "location": "1000",
        "shift": "Night",
        "checklist": list("INNNNIII"),
        "expectations": ["N/A", "No", "Yes", "Yes"],
    },
    3: {
        "location": "7000",
        "shift": "Night",
        "checklist": list("SSNNRISSI"),
        "expectations": ["No", "No", "Yes", "No"],
    },
}


@pytest.fixture(scope="session")
def sample_pages():
    return {page.number: page for page in load_pages(SAMPLE_PDF)}


@pytest.fixture(scope="session")
def page1(sample_pages):
    return sample_pages[1].image


@pytest.fixture(scope="session")
def aligned_pages(sample_pages):
    out = {}
    for n in (1, 2, 3):
        image = sample_pages[n].image
        out[n] = align_page(image, check_quality(image).skew_deg)
    return out


@pytest.fixture
def settings(tmp_path):
    # no key, temp cache
    return Settings(_env_file=None, gemini_api_key=None, cache_dir=tmp_path / "cache", max_workers=2)
