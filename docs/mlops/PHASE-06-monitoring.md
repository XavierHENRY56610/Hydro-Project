# Phase 06 — Monitoring & alerting

## Ce que ça apporte

Jusqu'ici le KGE n'était calculé qu'au moment de l'entraînement / promotion,
jamais sur la production au fil de l'eau, et aucune dérive n'était surveillée.
Cette phase ajoute : **dérive des données** (Evidently), **KGE en ligne**
(prévision archivée vs débit observé), **métriques Prometheus** métier +
techniques, **3 dashboards Grafana** provisionnés, et des **alertes**.

## Composants (`src/previ_r2d2/monitoring/`)

| Module | Rôle |
|---|---|
| `drift.py` | `run_drift_report(reference, current)` — Evidently `DataDriftPreset` ; **repli test KS/scipy** si la lib est absente ou son API a bougé. Seuil : ≥ 40 % de colonnes en dérive → dérive « dataset ». |
| `online_perf.py` | `evaluate_online(dossier, horizon)` — relit les `prevision_*.json` archivées des 30 derniers jours, aligne sur `debit_m3s` observé, calcule KGE / MAE / RMSE glissants (`kge_components`). |
| `state.py` | `outputs/monitoring/state.json` — écrit par le flow hebdo, lu par l'exporter (processus séparés). `write_demo_alert()` = `make fire-alert`. |
| `exporter.py` | Exporter Prometheus `:9200/metrics` — collector qui relit `state.json` + `version.json` + mtime des `prevision.json` à chaque scrape. |

## Flow Prefect

`weekly-monitoring` (`0 5 * * 1`, lundi 05h) — pour chaque `(dossier, horizon)`
promu : dérive (référence = `data_preparation.csv` figé avec le modèle,
courant = 7 derniers jours, rapport HTML dans `outputs/monitoring/`) + KGE en
ligne, le tout écrit dans `state.json`.

## Métriques exposées

**API** (`previ_api_*`, phase 03) : `requests_total`, `request_seconds`,
`predictions_total`, `model_kge`.
**Exporter** (`previ_*`) : `model_kge`, `model_version`, `online_kge`,
`online_rmse_m3s`, `data_drift_detected`, `data_drift_share`,
`hours_since_last_forecast`.

## Alertes (`infrastructure/monitoring/prometheus/rules/alert_rules.yml`)

| Alerte | Condition |
|---|---|
| `PreviApiDown` | `up{job="previ-api"} == 0` 2 min |
| `ModelKgeLow` | `previ_model_kge < 0.5` 10 min |
| `OnlineKgeDegraded` | `previ_online_kge < 0.3` 30 min |
| `DataDriftDetected` | `previ_data_drift_detected == 1` 5 min |
| `ForecastStale` | `previ_hours_since_last_forecast > 2` 15 min |
| `PreviExporterDown` | `up{job="previ-exporter"} == 0` 5 min |

## Dashboards Grafana (provisionnés)

`infrastructure/monitoring/grafana/dashboards/` :
- **API** — req/s par route, taux 5xx, latence P95, prévisions calculées
- **Modèle & Dérive** — KGE promu vs en ligne, RMSE en ligne, part de dérive, version
- **Infra** — CPU / mémoire / disque / charge (node-exporter)

## Docker

```bash
make monitor        # prometheus :9090 + grafana :3000 (admin/admin) + exporter :9200 + node-exporter
make fire-alert     # force KGE=-0.5 + drift=1 dans state.json -> alertes en firing
```

Services ajoutés : `prometheus`, `grafana`, `node-exporter`, `monitoring-exporter`.

## Tests

`tests/monitoring/` — 15 tests : dérive (pas de dérive / dérive détectée /
repli KS), KGE en ligne (parfait / biaisé / trop peu de points / lecture
archive+CSV), état (fusion, `write_demo_alert`), exporter (gauges émises).
Evidently testé via son repli quand l'API diffère.

## Ce qui reste

- Rapports Evidently servis dans une UI dédiée (`evidently ui` / Workspace) —
  pour l'instant HTML statiques dans `outputs/monitoring/`.
- Alerting Grafana natif (contact points Discord/Slack) : les règles
  Prometheus sont là ; le routage vers un webhook se branche en 1 fichier
  de provisioning quand l'URl est connue.
