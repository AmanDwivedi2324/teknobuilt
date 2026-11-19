# Common reader interface and result

from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np

from .schema import FormRead

ReadSource = Literal["model", "cache", "mock", "error"]


@dataclass
class ReadResult:
    source: ReadSource
    model: str | None
    data: FormRead
    error: str | None = None
    seconds: float = 0.0

    # true for real model replies
    @property
    def has_data(self) -> bool:
        return self.source in ("model", "cache")


class Reader(Protocol):
    def read(self, png: bytes) -> ReadResult: ...


# shrink image and encode as PNG
def encode_image(image: np.ndarray, max_width: int) -> bytes:
    scale = max_width / image.shape[1]
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode page image as PNG")
    return buf.tobytes()
