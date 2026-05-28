"""Storage abstraction — LocalStorage now, GCS swap-in later.

Don't add GCS until we actually need it; the Storage ABC exists so the swap
is a single new class implementing the same two methods, not a rewrite of
every call site that writes files.
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class Storage(ABC):
    """Two-method interface: write bytes, cleanup by age. Anything that
    speaks HTTP-fetchable bytes can implement this (local FS, GCS, S3)."""

    @abstractmethod
    def write(self, filename: str, data: bytes) -> str:
        """Persist bytes; return the public URL path callers can serve."""

    @abstractmethod
    def cleanup_older_than(self, age_hours: int) -> int:
        """Delete artifacts older than age_hours. Return count deleted."""


class LocalStorage(Storage):
    """Writes to a local directory; URL prefix is whatever the FastAPI app
    has mounted as a StaticFiles route (default '/outputs')."""

    def __init__(self, base_dir: str, url_prefix: str = "/outputs"):
        self.base_dir = base_dir
        self.url_prefix = url_prefix.rstrip("/")
        os.makedirs(base_dir, exist_ok=True)

    def write(self, filename: str, data: bytes) -> str:
        path = os.path.join(self.base_dir, filename)
        with open(path, "wb") as f:
            f.write(data)
        return f"{self.url_prefix}/{filename}"

    def cleanup_older_than(self, age_hours: int) -> int:
        cutoff = time.time() - (age_hours * 3600)
        deleted = 0
        base = Path(self.base_dir)
        if not base.exists():
            return 0
        for entry in base.iterdir():
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                try:
                    entry.unlink()
                    deleted += 1
                except OSError as e:
                    logger.warning("storage.cleanup: failed to delete %s: %s", entry, e)
        return deleted
