# Runs all steps on every page

import logging
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import __version__
from .alignment import align_page
from .checkboxes import read_checkboxes
from .config import Settings
from .extraction import Reader, encode_image
from .layout import DEFAULT_LAYOUT, FormLayout
from .models import ExtractionInfo, Issue, PageResult, RunResult, RunSummary, Severity, status_from_issues
from .pdf_loader import PdfLoadError, RenderedPage, load_pages
from .quality import check_quality
from .validation import build_form

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        reader: Reader,
        layout: FormLayout = DEFAULT_LAYOUT,
        debug_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.reader = reader
        self.layout = layout
        self.debug_dir = debug_dir

    # one page; never crash the run
    def process_page(self, page: RenderedPage) -> PageResult:
        try:
            return self._process_page(page)
        except Exception as exc:
            logger.exception("page %d failed", page.number)
            issue = Issue(
                code="processing_error", message=f"{type(exc).__name__}: {exc}", severity=Severity.REVIEW
            )
            return PageResult(page=page.number, status="needs_review", issues=[issue], form=None)

    def _process_page(self, page: RenderedPage) -> PageResult:
        if page.image is None:
            issue = Issue(
                code="render_failed",
                message=page.error or "page could not be rendered",
                severity=Severity.REJECT,
            )
            return PageResult(page=page.number, status="rejected", issues=[issue], form=None)

        # step 2: reject garbage early
        quality = check_quality(page.image, page.text_layer)
        if quality.rejected:
            logger.info("page %d rejected: %s", page.number, [i.code for i in quality.issues])
            return PageResult(
                page=page.number, status="rejected", issues=quality.issues, form=None, quality=quality.metrics
            )
        issues = list(quality.issues)

        # steps 3-4: align, read checkboxes
        alignment = align_page(page.image, quality.skew_deg, self.layout)
        pixel = None
        if alignment.ok:
            pixel = read_checkboxes(alignment.aligned, alignment.seen, self.layout)
        else:
            issues.append(
                Issue(
                    code="alignment_failed",
                    message=f"page could not be lined up with the form (score {alignment.score:.2f}); "
                    "checkboxes come from the model only",
                    severity=Severity.REVIEW,
                )
            )

        # step 5: read handwriting
        image_for_model = alignment.aligned if alignment.ok else page.image
        read = self.reader.read(encode_image(image_for_model, self.settings.llm_send_width_px))

        # step 6: validate and cross-check
        form, form_issues = build_form(pixel, read, self.layout)
        issues += form_issues

        if self.debug_dir is not None:
            self._save_debug_images(page, alignment.aligned)

        result = PageResult(
            page=page.number,
            status=status_from_issues(issues),
            issues=issues,
            form=form,
            quality=quality.metrics,
            alignment=alignment.as_dict(),
            extraction=ExtractionInfo(
                source=read.source, model=read.model, seconds=read.seconds, error=read.error
            ),
        )
        logger.info(
            "page %d: %s (%d issues, text from %s)", page.number, result.status, len(issues), read.source
        )
        return result

    def _save_debug_images(self, page: RenderedPage, aligned) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.debug_dir / f"page_{page.number:03d}.png"), page.image)
        if aligned is not None:
            cv2.imwrite(str(self.debug_dir / f"page_{page.number:03d}_aligned.png"), aligned)

    # whole PDF
    def run(self, pdf_path: Path) -> RunResult:
        errors: list[str] = []
        try:
            pages = self._process_all(load_pages(pdf_path, self.settings.render_dpi))
        except PdfLoadError as exc:
            logger.error("%s", exc)
            errors.append(str(exc))
            pages = []

        return RunResult(
            source_file=Path(pdf_path).name,
            processed_at=datetime.now(timezone.utc),
            pipeline_version=__version__,
            reader=self.settings.gemini_model
            if self.settings.has_llm
            else "mock (no API key; cache used if present)",
            summary=RunSummary(
                pages=len(pages),
                ok=sum(p.status == "ok" for p in pages),
                needs_review=sum(p.status == "needs_review" for p in pages),
                rejected=sum(p.status == "rejected" for p in pages),
            ),
            pages=pages,
            errors=errors,
        )

    # parallel pages, order kept, memory bounded
    def _process_all(self, pages: Iterable[RenderedPage]) -> list[PageResult]:
        results: list[PageResult] = []
        in_flight: deque[Future[PageResult]] = deque()
        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as pool:
            for page in pages:
                in_flight.append(pool.submit(self.process_page, page))
                if len(in_flight) >= 2 * self.settings.max_workers:
                    results.append(in_flight.popleft().result())
            while in_flight:
                results.append(in_flight.popleft().result())
        return results
