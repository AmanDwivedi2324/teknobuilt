# Step 1: render PDF pages to images

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

logger = logging.getLogger(__name__)


class PdfLoadError(Exception):
    pass


@dataclass
class RenderedPage:
    number: int
    image: np.ndarray | None  # None if render failed
    text_layer: str = ""
    error: str | None = None


# stream pages one at a time
def load_pages(pdf_path: Path, dpi: int = 200) -> Iterator[RenderedPage]:
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
            except Exception as exc:  # bad page, keep going
                logger.warning("page %d could not be rendered: %s", number, exc)
                yield RenderedPage(number, None, "", f"render failed: {exc}")
