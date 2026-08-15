"""Optional grouped raw HTTP debug output."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, TextIO


class DebugWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None
        self._lock = asyncio.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    async def write_conversation(self, value: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("debug writer is not started")
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        async with self._lock:
            self._handle.write(encoded + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
