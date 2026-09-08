"""Exporter Prometheus des métriques métier (phase 06).

Processus séparé de l'API : lit `state.json` (écrit par le flow
`weekly-monitoring`) + le système de fichiers (versions promues, fraîcheur
des prévisions) et expose le tout sur `:9200/metrics`. Prometheus scrape à
la fois l'API (`previ_api_*`, phase 03) et cet exporter (`previ_model_*`,
`previ_data_drift_*`, `previ_online_*`, `previ_hours_since_last_forecast`).
"""

from __future__ import annotations

import json
import logging
import time

from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily

from previ_r2d2.common import config
from previ_r2d2.monitoring import state

logger = logging.getLogger(__name__)

EXPORTER_PORT = 9200


def _iter_local_models():
    for version_path in sorted(config.MODELS_DIR.glob("*/h*/version.json")):
        horizon_dir = version_path.parent
        dossier = horizon_dir.parent.name
        horizon = horizon_dir.name.removeprefix("h")
        try:
            data = json.loads(version_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        yield dossier, horizon, data


def _hours_since_last_forecast(dossier: str) -> float | None:
    path = config.CENTRALES_DIR / dossier / "prevision.json"
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


class PreviCollector:
    """Custom collector : relit l'état à chaque scrape (pas de cache)."""

    def collect(self):
        kge = GaugeMetricFamily(
            "previ_model_kge", "KGE stacking du modèle promu (version.json)", labels=["dossier", "horizon"]
        )
        version = GaugeMetricFamily(
            "previ_model_version", "Numéro de version du modèle promu", labels=["dossier", "horizon"]
        )
        stale = GaugeMetricFamily(
            "previ_hours_since_last_forecast", "Heures depuis la dernière prévision écrite", labels=["dossier"]
        )
        online_kge = GaugeMetricFamily(
            "previ_online_kge", "KGE en ligne (prévision archivée vs débit observé)", labels=["dossier", "horizon"]
        )
        online_rmse = GaugeMetricFamily(
            "previ_online_rmse_m3s", "RMSE en ligne (m3/s)", labels=["dossier", "horizon"]
        )
        drift = GaugeMetricFamily(
            "previ_data_drift_detected", "1 si dérive des données détectée au dernier rapport", labels=["dossier", "horizon"]
        )
        drift_share = GaugeMetricFamily(
            "previ_data_drift_share", "Part des colonnes d'entrée en dérive", labels=["dossier", "horizon"]
        )

        seen_dossiers: set[str] = set()
        for dossier, horizon, data in _iter_local_models():
            seen_dossiers.add(dossier)
            if data.get("kge_stacking") is not None:
                kge.add_metric([dossier, horizon], float(data["kge_stacking"]))
            if data.get("version") is not None:
                version.add_metric([dossier, horizon], float(data["version"]))

        for skey, entry in state.read_all().items():
            dossier, _, htxt = skey.partition("/")
            horizon = htxt.removeprefix("h")
            seen_dossiers.add(dossier)
            if "online_kge" in entry:
                online_kge.add_metric([dossier, horizon], float(entry["online_kge"]))
            if "online_rmse" in entry:
                online_rmse.add_metric([dossier, horizon], float(entry["online_rmse"]))
            if "drift_detected" in entry:
                drift.add_metric([dossier, horizon], float(entry["drift_detected"]))
            if "drift_share" in entry:
                drift_share.add_metric([dossier, horizon], float(entry["drift_share"]))

        for dossier in sorted(seen_dossiers):
            hours = _hours_since_last_forecast(dossier)
            if hours is not None:
                stale.add_metric([dossier], hours)

        yield from (kge, version, stale, online_kge, online_rmse, drift, drift_share)


def serve(port: int = EXPORTER_PORT) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
    REGISTRY.register(PreviCollector())
    start_http_server(port)
    logger.info("exporter previ-R2-D2 sur :%d/metrics", port)
    while True:
        time.sleep(3600)


def main() -> None:
    import sys

    if "--fire-alert" in sys.argv:
        state.write_demo_alert()
        print("state.json : alerte de démo écrite (KGE=-0.5, drift=1)")
        return
    serve()


if __name__ == "__main__":
    main()
