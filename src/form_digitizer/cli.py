# Command line: python -m form_digitizer

import argparse
import logging
import sys
from pathlib import Path

from .config import get_settings
from .extraction import build_reader
from .pipeline import Pipeline


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="form-digitizer", description="Turn scanned safety observation forms into validated JSON."
    )
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("data/Sample_input.pdf"), help="input PDF")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output/results.json"), help="output JSON file"
    )
    parser.add_argument("--debug-dir", type=Path, help="also save each page and its aligned version here")
    parser.add_argument("--workers", type=int, help="pages processed in parallel (default from settings)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if args.workers:
        settings = settings.model_copy(update={"max_workers": args.workers})
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2", "google_genai"):  # hide HTTP request logs
        logging.getLogger(noisy).setLevel(logging.WARNING)

    result = Pipeline(settings, build_reader(settings), debug_dir=args.debug_dir).run(args.pdf)

    # save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # print short summary
    s = result.summary
    print(f"\n{result.source_file}: {s.pages} pages -> {s.ok} ok, ", end="")
    print(f"{s.needs_review} needs review, {s.rejected} rejected")
    for page in result.pages:
        codes = ", ".join(sorted({i.code for i in page.issues})) or "-"
        print(f"  page {page.page:>3}  {page.status:<13} {codes}")
    print(f"Written to {args.output}")
    return 2 if result.errors and not result.pages else 0


if __name__ == "__main__":
    sys.exit(main())
