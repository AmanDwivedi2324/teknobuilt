# No-key reader: reads nothing

from .base import ReadResult
from .schema import empty_read


class MockReader:
    def read(self, png: bytes) -> ReadResult:
        return ReadResult("mock", None, empty_read(), "no GEMINI_API_KEY configured; text fields not read")
