"""Disk cache in front of any reader: a page image is never paid for twice.

Cache files are committed with the repo, so anyone can re-run the pipeline without an API
key and get the same results the model gave.
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .base import Reader, ReadResult
from .schema import PROMPT_VERSION, FormRead

logger = logging.getLogger(__name__)


class CachedReader:
    def __init__(self, inner: Reader, cache_dir: Path, model: str, layout_name: str) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.model = model  # the model whose replies this cache holds
        self._namespace = f"{model}|{PROMPT_VERSION}|{layout_name}".encode()

    def path_for(self, png: bytes) -> Path:
        digest = hashlib.sha256(png + self._namespace).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def read(self, png: bytes) -> ReadResult:
        path = self.path_for(png)
        if path.exists():
            try:
                return ReadResult("cache", self.model, FormRead.model_validate_json(path.read_text("utf-8")))
            except (ValidationError, OSError) as exc:
                logger.warning("ignoring broken cache entry %s: %s", path.name, exc)
                path.unlink(missing_ok=True)

        result = self.inner.read(png)
        if result.source == "model":  # only real replies are worth keeping
            self._write(path, result.data)
        return result

    def _write(self, path: Path, data: FormRead) -> None:
        """Write to a temp file and rename, so a crash never leaves half a cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=2))
        os.replace(tmp, path)
