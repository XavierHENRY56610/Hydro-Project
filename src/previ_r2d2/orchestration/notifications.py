"""Notification d'échec de flow (phase 05).

Remplace le digest mail retiré. Si `PREFECT_FAILURE_WEBHOOK` est défini
(URL Discord ou Slack entrante), un flow qui finit en échec poste un message
court. Sans la variable : no-op silencieux (dev / CI / tests).

Hub'Eau tombe souvent → les tâches réseau ont déjà des retries ; la
notification n'arrive donc qu'après épuisement des tentatives.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)


def _webhook_url() -> str:
    return os.environ.get("PREFECT_FAILURE_WEBHOOK", "").strip()


def _payload(text: str) -> bytes:
    # Slack : {"text": ...} ; Discord : {"content": ...}. On envoie les deux
    # clés, chaque service ignore celle qu'il ne connaît pas.
    return json.dumps({"text": text, "content": text}).encode("utf-8")


def notify_failure(flow, flow_run, state) -> None:
    """Hook `on_failure` d'un flow Prefect."""
    url = _webhook_url()
    name = getattr(flow_run, "name", "?")
    flow_name = getattr(flow, "name", getattr(flow, "__name__", "flow"))
    message = f":rotating_light: previ-R2-D2 — flow `{flow_name}` en échec (run `{name}`)."
    logger.error(message)
    if not url:
        return
    try:
        req = urllib.request.Request(
            url, data=_payload(message), headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001 — une notif ratée ne doit pas masquer l'échec du flow
        logger.warning("notification d'échec non envoyée : %s", exc)
