"""Structured JSON logging. Every component logs through this so redaction
runs on log output too — logs must never leak what the DB wouldn't (§4.5).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_REDACTOR_HOOK: Any = None


def _get_redactor() -> Any:
    global _REDACTOR_HOOK
    if _REDACTOR_HOOK is None:
        from shadowtrace.proxy.redact import scrub_text

        _REDACTOR_HOOK = scrub_text
    return _REDACTOR_HOOK


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = record.correlation_id
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        redactor = _get_redactor()
        scrubbed_message, _flags = redactor(payload["message"])
        payload["message"] = scrubbed_message
        return json.dumps(payload, default=str)


def get_logger(
    component: str, *, log_dir: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(f"shadowtrace.{component}")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_dir / f"{component}.log", maxBytes=10_000_000, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
