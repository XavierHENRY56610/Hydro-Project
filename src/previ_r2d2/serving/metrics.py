"""Métriques Prometheus de l'API (phase 03).

Périmètre volontairement réduit : métriques *techniques* de l'API (requêtes,
latence, erreurs) + une gauge métier `previ_model_kge` alimentée à chaque
prévision. Les métriques de dérive et de KGE en ligne arrivent en phase 06
(monitoring), où Evidently + Prometheus + Grafana sont câblés proprement.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "previ_api_requests_total", "Requêtes HTTP traitées", ["method", "path", "status"]
)
LATENCY = Histogram(
    "previ_api_request_seconds", "Latence des requêtes HTTP", ["method", "path"]
)
PREDICTIONS = Counter(
    "previ_api_predictions_total", "Prévisions calculées", ["dossier", "horizon", "outcome"]
)
MODEL_KGE = Gauge(
    "previ_model_kge", "KGE stacking du modèle servi (dernière prévision)", ["dossier", "horizon"]
)


def render() -> tuple[bytes, str]:
    """Corps + content-type pour `GET /metrics`."""
    return generate_latest(), CONTENT_TYPE_LATEST
