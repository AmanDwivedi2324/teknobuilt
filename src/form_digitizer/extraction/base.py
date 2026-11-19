"""The interface every text reader implements, so providers can be swapped freely."""

from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np

from .schema import FormRead

ReadSource = Literal["model", "cache", "mock", "error"]


@dataclass
class ReadResult:
    source: ReadSource  # model: fresh reply; cache: saved reply; mock: no model; error: call failed
    model: str | None
    data: FormRead
    error: str | None = None
    seconds: float = 0.0

    @property
    def has_data(self) -> bool:
        """True when the fields came from a real model reply (fresh or cached)."""
        return self.source in ("model", "cache")


class Reader(Protocol):
    def read(self, png: bytes) -> ReadResult:
        """Read one page image (PNG bytes). Must never raise."""
        ...


def encode_image(image: np.ndarray, max_width: int) -> bytes:
    """Shrink to max_width (fewer image tokens) and encode as PNG."""
    scale = max_width / image.shape[1]
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode page image as PNG")
    return buf.tobytes()
