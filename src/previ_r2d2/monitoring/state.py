"""État de monitoring partagé (phase 06).

Un seul fichier JSON `outputs/monitoring/state.json`, clé `<dossier>/h<horizon>`.
Écrit par le flow `weekly-monitoring`, lu par l'exporter Prometheus (processus
séparé) — d'où le passage par un fichier plutôt qu'un registre en mémoire.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from previ_r2d2.common import config


def _path() -> Path:
    # Recalculé à chaque appel : les tests monkeypatchent config.ROOT.
    return config.ROOT / "outputs" / "monitoring" / "state.json"


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def key(dossier: str, horizon: int) -> str:
    return f"{dossier}/h{horizon}"


def update(dossier: str, horizon: int, **metrics: float) -> None:
    """Fusionne `metrics` dans l'entrée `(dossier, horizon)` et horodate."""
    data = _load()
    entry = data.get(key(dossier, horizon), {})
    entry.update(metrics)
    entry["updated_at"] = time.time()
    data[key(dossier, horizon)] = entry
    _save(data)


def read_all() -> dict:
    return _load()


def write_demo_alert() -> None:
    """`make fire-alert` — force un KGE effondré + une dérive sur toutes les
    entrées connues (ou une entrée factice si l'état est vide), pour la démo."""
    data = _load() or {key("demo", 8): {}}
    now = time.time()
    for entry in data.values():
        entry.update(online_kge=-0.5, drift_detected=1, drift_share=1.0, updated_at=now)
    _save(data)
