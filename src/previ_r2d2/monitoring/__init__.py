"""Monitoring & alerting previ-R2-D2 (phase 06).

- `drift.py`        : rapport de dérive (Evidently `DataDriftPreset`, repli KS/scipy).
- `online_perf.py`  : KGE / MAE / RMSE glissants — prévision archivée vs débit observé.
- `state.py`        : état partagé JSON (écrit par le flow hebdo, lu par l'exporter).
- `exporter.py`     : exporter Prometheus (`:9200/metrics`) — gauges métier.

Le flow Prefect `weekly-monitoring` (phase 05) calcule dérive + perf en ligne
et met `state.json` à jour ; Prometheus scrape l'exporter ; Grafana affiche
3 dashboards provisionnés (`infrastructure/monitoring/`).
"""
