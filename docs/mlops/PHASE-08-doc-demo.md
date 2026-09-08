# Phase 08 — Documentation, démo, bonus

## Documentation

| Fichier | Contenu |
|---|---|
| `ARCHITECTURE.md` (racine) | vue d'ensemble + diagramme Mermaid, composants, cadences, ports |
| `MLOPS.md` (racine) | **chaque brique → le cours DataScientest dont elle vient** (obligatoire / optionnel / écarté), doubles filets assumés, **rétro « v2 » de simplification** |
| `docs/mlops/ORCHESTRATORS.md` | comparaison Prefect (retenu) vs Airflow vs ZenML |
| `docs/mlops/PHASE-0X-*.md` | une fiche par phase (ce qui est construit, comment le lancer, ce qui reste) |
| `SETUP.md` | démarrage en 3 commandes + table des cibles `make` |

## Démo bout-en-bout

```bash
make demo
```

`cron/scripts/demo.py` (dans le conteneur `trainer`, MLflow branché) :

1. fabrique une centrale synthétique `demo_centrale` (`data_preparation.csv`,
   `bv.json`, entrée `config-general.json`) ;
2. entraînement rapide (`epochs=2`, `n_trials=0`) + run + enregistrement au
   Model Registry ;
3. **promotion** : `models/demo_centrale/h72/` + tag git `demo_centrale-h72-v1`
   + alias MLflow `@production` (⚠️ crée un commit + tag locaux, comme
   `promotion.py` en usage normal) ;
4. **prédiction servie** depuis le modèle résolu (`registry` → repli `local`) ;
5. écrit un état de monitoring (`state.json`).

`make demo` démarre ensuite API + monitoring + nginx et affiche les URLs.
`make fire-alert` force une dérive → alertes `firing`.

## Bonus — orchestrateurs alternatifs

`infrastructure/orchestration-alternatives/` (non branché sur l'image / la CI) :

- **Airflow** (`airflow/`) — cours obligatoire S12. 4 DAGs équivalents aux
  flows Prefect (`BashOperator` → mêmes `cron/scripts/`), compose Airflow
  dédié. `docker compose -f .../docker-compose.airflow.yml up` → UI :8081.
- **ZenML** (`zenml/`) — cours optionnel S14. Pipeline d'entraînement en
  `@pipeline`/`@step` avec `ArtifactConfig` (versionnage natif). `pip install
  "zenml[server]"` puis `python .../training_pipeline.py`.

## Reste pour une vraie mise en production

Voir la rétro dans `MLOPS.md` — cible minimale défendable : **DVC + MLflow +
FastAPI + Prefect + Grafana + Evidently + CI**. Le reste (MinIO, Kubernetes,
node-exporter, Airflow/ZenML) est pédagogique et se retire proprement.

Autres pistes : API météo publique (le trou connu du projet), feature store,
canary / A-B sur les modèles, backfill historique, alerting Grafana routé
vers un webhook Discord/Slack.
