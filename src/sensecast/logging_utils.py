"""Structured JSON logging shared by the pipeline, API and dashboard."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "sensecast") -> logging.Logger:
    logger = logging.getLogger(name)
    root = logging.getLogger("sensecast")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


@contextmanager
def timed(logger: logging.Logger, step: str, **fields):
    """Log start/end of a pipeline step with its duration in seconds."""
    t0 = time.perf_counter()
    logger.info("step_start", extra={"step": step, **fields})
    try:
        yield
    finally:
        logger.info(
            "step_end",
            extra={"step": step, "seconds": round(time.perf_counter() - t0, 3), **fields},
        )
