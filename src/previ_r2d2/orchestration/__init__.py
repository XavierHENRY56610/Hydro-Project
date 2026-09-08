"""Orchestration Prefect (phase 05).

`flows.py` enrobe les scripts `cron/scripts/` (déjà testés, lancés jusqu'ici
à la main ou par `dvc repro`) en `@task` / `@flow` Prefect, avec retries et
notifications d'échec. `serve.py` publie les 3 déploiements planifiés
(horaire / quotidien / mensuel) — c'est le processus du conteneur
`prefect-worker`.

Aucune logique métier ici : les corps de tâche restent les scripts existants,
exécutés en sous-processus (même invocation que les stages DVC).
"""
