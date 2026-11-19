"""Step 5: read handwriting with a vision-language model (or a stand-in when there's no key)."""

import logging

from ..config import Settings
from ..layout import DEFAULT_LAYOUT, FormLayout
from .base import Reader, ReadResult, encode_image
from .cache import CachedReader
from .mock import MockReader

logger = logging.getLogger(__name__)

__all__ = ["Reader", "ReadResult", "build_reader", "encode_image"]


def build_reader(settings: Settings, layout: FormLayout = DEFAULT_LAYOUT) -> CachedReader:
    """Gemini when a key is configured, otherwise the mock; both behind the disk cache."""
    if settings.gemini_api_key is not None:
        from .gemini import GeminiReader  # imported only when needed

        inner: Reader = GeminiReader(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            layout=layout,
            max_attempts=settings.llm_max_attempts,
            timeout_seconds=settings.llm_timeout_seconds,
            max_concurrency=settings.llm_max_concurrency,
        )
    else:
        logger.warning(
            "GEMINI_API_KEY is not set: using cached replies where available, otherwise text is not read"
        )
        inner = MockReader()
    return CachedReader(inner, settings.cache_dir, settings.gemini_model, layout.name)
