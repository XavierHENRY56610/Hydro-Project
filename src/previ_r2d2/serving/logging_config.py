"""Logs JSON structurés + identifiant de requête (phase 03).

Un log = une ligne JSON (timestamp, level, logger, message, + champs
contextuels). L'`request_id` (en-tête `X-Request-ID` entrant ou UUID généré)
est propagé via un `ContextVar` et injecté dans chaque enregistrement émis
pendant le traitement de la requête.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Formate chaque enregistrement en une ligne JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Remplace les handlers racine par un unique handler JSON sur stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn tient ses propres loggers : on les laisse remonter à la racine.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
