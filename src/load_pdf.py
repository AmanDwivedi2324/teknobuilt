"""Step 1: turn each PDF page into a grayscale image and save any text the page contains."""
import json
import sys
from pathlib import Path

import pymupdf as fitz

DPI = 200  # sharp enough for X marks and handwriting; higher only slows things down


def render_pages(pdf_path: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # a corrupted or non-PDF file must not crash the run
        print(f"Could not open {pdf_path}: {exc}")
        return pages

    for i, page in enumerate(doc, start=1):
        info = {"page": i, "image": None, "size": None, "text_layer": "", "error": None}
        try:
            pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
            img_path = out_dir / f"page_{i:02d}.png"
            pix.save(img_path)
            info["image"] = str(img_path)
            info["size"] = [pix.width, pix.height]
            info["text_layer"] = page.get_text().strip()
        except Exception as exc:  # one bad page is recorded, the others still run
            info["error"] = str(exc)
        pages.append(info)

    return pages


if __name__ == "__main__":
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "data/Sample_input.pdf")
    print(json.dumps(render_pages(pdf, Path("output/pages")), indent=2))