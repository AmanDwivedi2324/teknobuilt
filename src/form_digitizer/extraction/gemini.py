"""Text reader backed by Google Gemini (works on the free tier)."""

import logging
import random
import threading
import time
from collections.abc import Callable

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from ..layout import FormLayout
from .base import ReadResult
from .schema import SYSTEM_PROMPT, USER_PROMPT, FormRead, build_response_schema, empty_read

logger = logging.getLogger(__name__)

BACKOFF_BASE_SECONDS = 5.0  # waits 5, 10, 20 ... s (plus jitter) between attempts
BACKOFF_MAX_SECONDS = 60.0


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, so parallel workers don't retry in lockstep."""
    return min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)) + random.uniform(0, 1)


class GeminiReader:
    def __init__(
        self,
        api_key: str,
        model: str,
        layout: FormLayout,
        max_attempts: int = 4,
        timeout_seconds: float = 120.0,
        max_concurrency: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._slots = threading.Semaphore(max_concurrency)  # caps parallel API calls across workers
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000))
        )
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=build_response_schema(layout),
            temperature=0,  # same image in, same answer out
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # no tools used
        )

    def _call(self, png: bytes) -> str:
        with self._slots:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), USER_PROMPT],
                config=self._config,
            )
        return response.text or ""

    def read(self, png: bytes) -> ReadResult:
        start = time.monotonic()
        error = "no attempt made"
        for attempt in range(1, self._max_attempts + 1):
            try:
                data = FormRead.model_validate_json(self._call(png))  # reject replies that break the schema
                return ReadResult("model", self.model, data, None, round(time.monotonic() - start, 2))
            except ValidationError as exc:
                error = f"reply did not match the schema ({exc.error_count()} problems)"
            except errors.ClientError as exc:
                error = f"API error {exc.code}: {str(exc)[:200]}"
                if exc.code != 429:  # bad key or bad request: retrying will not help
                    break
            except Exception as exc:  # server errors, timeouts, network problems
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("Gemini attempt %d/%d failed: %s", attempt, self._max_attempts, error)
            if attempt < self._max_attempts:
                self._sleep(backoff_seconds(attempt))
        return ReadResult("error", self.model, empty_read(), error, round(time.monotonic() - start, 2))
