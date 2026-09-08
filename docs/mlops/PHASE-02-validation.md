# Phase 02 — Validation des données

## Ce que ça apporte

Un **contrat explicite** sur `data_preparation.csv`, vérifié à deux endroits :

1. **Stage DVC `validate`** (`dvc/preprocessing/dvc.yaml`), intercalé entre
   `data_preparation` et l'entraînement — une erreur bloquante arrête le
   pipeline (`dvc repro` échoue) avant qu'un modèle biaisé ne soit entraîné.
2. **Dans `run_training`** (`orchestrator.py`) juste après `load_df` —
   `strict=True`, lève `DataValidationError` avec un message clair.

Le résultat de la validation est **tracé dans MLflow** (tag `data_validation_ok`,
`data_validation_warnings`, + les warnings en artefact `data_validation/`).

## Le contrat (`preprocessing/data_preparation/validation.py`)

| Contrôle | Niveau |
|---|---|
| DataFrame non vide | erreur |
| index `DatetimeIndex`, tz-naive, trié, sans doublon | erreur |
| colonne cible `debit_m3s` présente et non entièrement vide | erreur |
| `debit_m3s` / `debit_amont*` / `precipitation*` sans valeur négative | erreur |
| trou de débit > `MAX_DEBIT_GAP_HOURS` (168 h) | warning |
| historique de débit < `MIN_HISTORY_DAYS` (365 j) | warning |
| `temperature*` hors plage Kelvin plausible [180, 340] | warning |

Pas de `pandera` : les colonnes varient par centrale (amont / points météo) et
on veut distinguer erreurs bloquantes et warnings — plus lisible en code direct.

Les colonnes météo restent **figées** (acquisition FTP retirée) → leur absence
ou leurs NaN ne sont **pas** des erreurs.

## Utilisation

```bash
# stage seul
docker compose run --rm trainer python cron/scripts/validate-data.py
docker compose run --rm trainer python cron/scripts/validate-data.py --dossier touzac_g2_G2

# via le pipeline (validate tourne après data_preparation)
docker compose run --rm trainer dvc repro dvc/preprocessing/dvc.yaml
```

`train_new` / `train_monthly` (`dvc/model/dvc.yaml`) dépendent maintenant du
marqueur `validate.json` : l'entraînement ne part pas si la validation a échoué.

## Fichiers touchés

- `src/previ_r2d2/preprocessing/data_preparation/validation.py` — **nouveau**
- `cron/scripts/validate-data.py` — **nouveau**, stage DVC
- `dvc/preprocessing/dvc.yaml` — stage `validate` ; `dvc/model/dvc.yaml` — dép ajoutée
- `src/previ_r2d2/model/pipeline/orchestrator.py` — validation avant entraînement
- `src/previ_r2d2/model/tracking/mlflow_tracking.py` — `log_data_validation`

## Ce qui reste pour la phase 06

Le **jeu de référence de dérive** (fenêtre d'entraînement de chaque modèle
promu) et le **rapport de qualité Evidently** seront ajoutés avec le monitoring
(phase 06), où Evidently est mis en place proprement.
