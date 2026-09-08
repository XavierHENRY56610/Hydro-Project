"""Publie les 3 déploiements planifiés et les sert — processus du conteneur
`prefect-worker` (phase 05).

`prefect serve` interroge le serveur Prefect, déclenche les runs planifiés
et les exécute dans des sous-processus. Cadences alignées sur le README :
horaire (prévision), quotidien 03h30 (pipeline + entraînement des nouvelles
centrales), mensuel le 1er à 02h00 (réentraînement + promotion).
"""

from __future__ import annotations

from prefect import serve

from previ_r2d2.orchestration.flows import (
    daily_pipeline_flow,
    hourly_forecast_flow,
    monthly_retrain_flow,
    weekly_monitoring_flow,
)

CRON_HOURLY = "7 * * * *"       # à hh:07, après l'écriture des débits de l'heure
CRON_DAILY = "30 3 * * *"      # 03h30
CRON_WEEKLY = "0 5 * * 1"      # lundi 05h00 (dérive + perf en ligne)
CRON_MONTHLY = "0 2 1 * *"     # le 1er du mois à 02h00


def main() -> None:
    serve(
        hourly_forecast_flow.to_deployment(name="hourly-forecast", cron=CRON_HOURLY),
        daily_pipeline_flow.to_deployment(name="daily-pipeline", cron=CRON_DAILY),
        weekly_monitoring_flow.to_deployment(name="weekly-monitoring", cron=CRON_WEEKLY),
        monthly_retrain_flow.to_deployment(name="monthly-retrain", cron=CRON_MONTHLY),
    )


if __name__ == "__main__":
    main()
