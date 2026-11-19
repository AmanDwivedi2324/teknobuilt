"""Step 1: render each PDF page to a grayscale image, keeping any text the page carries."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

logger = logging.getLogger(__name__)


class PdfLoadError(Exception):
    """The file could not be opened as a document at all."""


@dataclass
class RenderedPage:
    number: int  # 1-based
    image: np.ndarray | None  # 2-D uint8 grayscale; None if this page failed to render
    text_layer: str = ""  # scanned pages have none; digital stamps such as "FAILED TEST FORM" do
    error: str | None = None


def load_pages(pdf_path: Path, dpi: int = 200) -> Iterator[RenderedPage]:
    """Yield pages one at a time so a long PDF never has to sit in memory all at once.

    Raises PdfLoadError if the file cannot be opened. A single broken page is yielded
    with image=None and an error message instead of stopping the whole file.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise PdfLoadError(f"cannot open {pdf_path}: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise PdfLoadError(f"{pdf_path} is password-protected")

    with doc:
        for number, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
                rows = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
                image = rows[:, : pix.width].copy()
                yield RenderedPage(number, image, page.get_text().strip())
            except Exception as exc:  # one bad page is recorded; the rest still run
                logger.warning("page %d could not be rendered: %s", number, exc)
                yield RenderedPage(number, None, "", f"render failed: {exc}")
