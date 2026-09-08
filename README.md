# previ-R2-D2 (pipeline IA)

> **Version projet de cours MLOps** — ce dépôt est une version réduite de
> previ-R2-D2 (pipeline de production chez Barthe EnR), adaptée pour un
> projet de cours réalisé hors du réseau de l'entreprise. Périmètre réduit
> à 3 centrales (`apas_G1_G4`, `nancy_A`, `touzac_g2_G2`, toutes
> `flex_strategy: DEFAULT`). Les modules suivants, inaccessibles hors
> serveur de production, ont été retirés :
> - **OneGate** (`memorandum.py`) — API interne de structure des centrales ;
>   `config-general.json`/`config-raccordement.json` des 3 centrales sont
>   désormais des données statiques, versionnées via DVC.
> - **hydrospot_stream** (`preprocessing/puissance/`) — source de puissance
>   sur NAS ; `puissance.csv`/`puissance_horaire.csv` figés, idem.
> - **FTP météo NWP** (`nwp_ftp.py`/`retention.py`) — seul `nwp_reader.py`
>   (parseur pur, sans réseau) est conservé ; les colonnes météo de
>   `data_preparation.csv` restent figées en attendant un sous-projet
>   séparé de remplacement par une API météo publique.
> - **automate** (rsync/SSH, `HAUTE_CHUTE`) — aucune des 3 centrales
>   gardées n'utilise cette stratégie.
> - **Mail/digest quotidien** — remplacé par une notification webhook
>   Discord/Slack sur échec de flow Prefect (phase 05).
> - **MLflow** — retiré puis **reconstruit proprement** dans le cadre du
>   projet MLOps (phase 01 : tracking + Model Registry).
>
> **Couche MLOps ajoutée** (projet de cours, 8 phases) : socle Docker,
> MLflow + Model Registry, validation des données, API FastAPI, CI/CD
> GitHub Actions, orchestration Prefect, monitoring Evidently +
> Prometheus/Grafana, déploiement Kubernetes (Helm) + Nginx.
> → **[`ARCHITECTURE.md`](ARCHITECTURE.md)** · **[`MLOPS.md`](MLOPS.md)**
> (brique → cours) · **[`SETUP.md`](SETUP.md)** ·
> [`docs/mlops/`](docs/mlops/) (une fiche par phase).
>
> Reste pleinement fonctionnel : le débit (Hub'Eau/eaufrance, API
> publique), l'onboarding BV, l'entraînement et la prédiction (mode
> `source="frozen"` disponible pour une prédiction 100% reproductible sans
> réseau, en plus du mode `source="live"` par défaut).

Pipeline IA de prévision hydrologique : collecte des débits (eaufrance /
Hub'Eau), caractérisation du bassin versant, entraînement et prédiction du
modèle hybride (LightGBM + BiLSTM + stacking), orchestrés via DVC.

## Architecture

```
previ-R2-D2/
├── pyproject.toml               # package installable (pip install -e .)
├── requirements.txt
├── run.py                        # CLI entraînement/prédiction du modèle hybride (--train ...)
├── config/
│   ├── centrales/                # réservé (vide, .gitkeep) -- inutilisé dans cette version
│   ├── bv_mapping.yaml          # dossier -> nom de shapefile BV (centrales/REFERENCE/shapefiles/)
│   └── puissance_mapping.yaml   # dossier -> nom de dossier hydrospot_stream (repli explicite, cf. ci-dessous)
├── src/previ_r2d2/
│   ├── common/                  # config, secret_config (3 secrets restants), dvc_markers
│   ├── preprocessing/
│   │   ├── onboarding/          # validation.py -- complétude config-raccordement.json avant bv
│   │   ├── debit/                # eaufrance, Hub'Eau, stockage local (centrales/<dossier>/) + dédup
│   │   ├── puissance/            # puissance_store.py seul : résolution du dossier hydrospot_stream
│   │   │                        #   (mapping/heuristique, utilisée par la validation d'onboarding) --
│   │   │                        #   import/fusion réels (NAS) retirés, puissance*.csv figés
│   │   ├── meteo/                 # nwp_reader.py seul (parseur, FTP retiré)
│   │   ├── data_preparation/      # Data_Preparation : débit + météo NWP brute + amont brut
│   │   └── bv/                   # onboarding BV : bassin versant, stations hydrométriques, transit
│   ├── model/
│   │   ├── features/             # et0, snow, meteo_hydro, debit_autoregressif, amont —
│   │   │                         #   feature engineering du modèle hybride, port fidèle de feature()
│   │   ├── architectures/
│   │   │   ├── lightgbm/          # metrics, features, training (Optuna+OOF+final), predict, explain (SHAP)
│   │   │   ├── bilstm/            # model.py (BiLSTMHydro, torch), sequences, metrics, explain (attention)
│   │   │   └── stacking.py        # meta-learner Ridge/LGBM (fonctions pures)
│   │   └── pipeline/              # orchestration cross-architecture :
│   │                              #   bv_config, data_loading, oof_cache, stacking_fit, metrics,
│   │                              #   predict, artifacts, plots, orchestrator (run_training),
│   │                              #   predict_orchestrator (run_prediction), predict_window,
│   │                              #   eligibility (12 mois / réentraînement mensuel),
│   │                              #   promotion (comparaison KGE + promotion versionnée DVC+git)
│   ├── cli.py                     # point d'entrée de run.py (--train/--predict, MANUEL uniquement)
│   └── postprocessing/           # archive.py (archivage horaire, local ./ARCHIVE) -- API FastAPI Previ_v2 non portée
├── models/                        # SOURCE DE VÉRITÉ modèles en PRODUCTION, versionné DVC+git
│                                  #   <dossier>/h<horizon>/{version.json, meta_config.json, bilstm.pt, ...}
├── weights/
│   ├── hybrid/<dossier>/h<horizon>/           # zone de travail manuelle (run.py, expés)
│   └── hybrid_candidate/<dossier>/h<horizon>/ # candidat en cours d'évaluation par train.py (auto)
├── ARCHIVE/                        # archive locale des prévisions horaires (gitignored)
├── cron/
│   └── scripts/                  # CLI minces (maj-data, onboarding-check, onboarding-bv,
│                                  #   build-data-preparation, train, predict-archive) --
│                                  #   lancement manuel, pas de cron/wrappers/ (retiré, cf. note en tête)
├── dvc/
│   ├── preprocessing/dvc.yaml     # debit -> onboarding_check -> bv -> data_preparation (manuel)
│   ├── model/dvc.yaml             # train_new (quotidien, nouvelles centrales), train_monthly (mensuel)
│   └── postprocessing/dvc.yaml    # predict_archive (horaire)
├── outputs/                       # sorties de prédiction/entraînement (gitignored)
├── tests/                         # miroir de src/previ_r2d2/
├── centrales/                     # données des 3 centrales -- <dossier>/ versionné via DVC
│   │                              #   (remote local ../remote_dvc, cf. <dossier>.dvc à la racine)
│   ├── REFERENCE/                 # bv_rules.json + centrales_calibration.json (git-tracké) ;
│   │                              #   config-general.json + shapefiles/ versionnés via DVC ;
│   │                              #   files/ (gitignoré, non utilisé dans cette version)
│   └── <dossier>/                 # config-raccordement.json, *.csv, bv.json, data_preparation.csv,
│                                  #   prevision.json, enchere.json
└── data/                          # inutilisé pour l'instant (réservé à un usage futur)
```

## Installation

**Projet à deux — tout tourne dans Docker.** Aucun Python / conda à installer.

```bash
cp .env.example .env          # renseigner DAGSHUB_USER + DAGSHUB_TOKEN
make build                    # construit l'image de base
make up
docker compose run --rm trainer dvc pull   # données + modèles des 3 centrales
make test                     # ~390 tests
```

Détail complet, cibles Make et travail en équipe : **[`SETUP.md`](SETUP.md)**.

<details>
<summary>Ancienne installation conda locale (fallback hors Docker)</summary>

```bash
conda create -n projet-mlops python=3.11 -c conda-forge -y
conda activate projet-mlops
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" --no-deps
pip install -r infrastructure/docker/requirements.in
dvc pull
```

`rasterio`/`geopandas`/`pysheds` (extra `gis`) restent volontairement non
installés — lazy-import only dans `delineation.py`, jamais exercés (les 3
centrales ont toutes un shapefile connu). Le shim `.venv/bin/python` n'est
plus référencé (`dvc/*/dvc.yaml` appelle `python` directement, résolu dans
le conteneur).
</details>

## Travail en équipe (DagsHub)

Le dépôt (code + données) est partagé via [DagsHub](https://dagshub.com/Sebastien6631/Hydro-Projet)
— un seul endroit pour le git et le remote DVC.

```bash
git clone https://dagshub.com/Sebastien6631/Hydro-Projet.git
cd Hydro-Projet
dvc pull   # récupère les données (config-general.json, shapefiles/,
           # centrales/<dossier>/..., modèles entraînés)
```

**Configuration une fois par personne** : chacun crée son propre token
DagsHub (Settings → Tokens sur dagshub.com), puis :
```bash
dvc remote modify dagshub --local user <votre_pseudo_dagshub>
dvc remote modify dagshub --local password <votre_token>
```
(`.dvc/config.local` est gitignoré — jamais commité/partagé.)

**Après un entraînement ou une nouvelle donnée** : `promote_model` fait un
`git commit`/`git tag` **local uniquement**. Pour que l'équipe le
récupère, il faut pousser les deux à la main :
```bash
git push origin <branche>
dvc push
```

## Configuration

Les secrets vivent dans `src/previ_r2d2/common/secret_config.py` (Python
local, **non versionné**) :

```python
PREVI_PUISSANCE_SOURCE_ROOT = "..." # racine hydrospot_stream (utilisée seulement pour la résolution
                                     #   du mapping puissance -- import/fusion réels retirés)
PREVI_MNT = "..."                   # GeoTIFF MNT France entière (repli délimitation BV)
PREVI_NAS_METEO = "..."             # racine des fichiers météo NWP bruts (acquisition FTP retirée --
                                     #   dossier vide/absent = colonnes météo vides, dégradation gracieuse)
```

Priorité de lecture : `secret_config.py` > variables d'environnement > défauts. Les secrets
propres aux modules retirés dans cette version (cf. note en tête de fichier) ont disparu
avec eux.

## Pipeline (DVC)

```bash
dvc dag dvc/preprocessing/dvc.yaml       # debit -> onboarding_check -> bv -> data_preparation (manuel)
dvc dag dvc/model/dvc.yaml               # train_new (quotidien), train_monthly (mensuel)
dvc dag dvc/postprocessing/dvc.yaml      # predict_archive (horaire)
dvc repro dvc/preprocessing/dvc.yaml     # exécute tout ce pilier, dans l'ordre
```

```
debit ──> onboarding_check ──> bv
                                 │
                                 ▼
              (data_preparation : manuel uniquement,
               train.py le rafraîchit lui-même par dossier)
                                 │
       dvc/model/dvc.yaml : train_new  ──┐
       dvc/model/dvc.yaml : train_monthly┤
                                          │
       dvc/postprocessing/dvc.yaml : predict_archive
```

`debit`, `onboarding_check` et `bv` déclarent chacun un `outs:` minimal
(`logs/dvc_markers/<stage>.json`, `cache: false`) — pas une vraie sortie mise
en cache, juste un marqueur horodaté écrit en une ligne
(`previ_r2d2.common.dvc_markers.write(...)`) à la fin de chaque script, pour
donner une vraie arête DAG entre stages (sans ça, DVC n'a rien à quoi
accrocher une dépendance). `debit` et `onboarding_check` gardent en plus
`always_changed: true` (comme `data_preparation` plus bas) — leurs
dépendances déclarées ne suffisent pas à elles seules à détecter un
changement réel (ex. nouveau point Hub'Eau), donc l'exécution est forcée à
chaque `dvc repro`.
`data_preparation` **n'est appelé par rien d'automatisé** — `train.py`
(`dvc/model/dvc.yaml`) le rafraîchit lui-même, dossier par dossier, juste
avant de vérifier l'éligibilité de CE dossier (décision actée : un `foreach`
DVC par dossier aurait nécessité une liste de dossiers maintenue hors du yaml
+ restructurer `train` en stages par dossier pour un lien réellement
significatif — jugé disproportionné). Le stage reste un outil manuel
(`dvc repro dvc.yaml:data_preparation`, backfill groupé), lui aussi
`always_changed: true` (les colonnes débit/amont sont réellement
rafraîchies ; les colonnes météo restent figées, cf. note en tête de fichier).

## Les scripts

Les raccordements (`config-general.json` + `config-raccordement.json` par
dossier) sont désormais des données statiques versionnées via DVC pour les 3
centrales gardées — il n'y a plus de script pour les régénérer depuis une API
externe (cf. note en tête de fichier).

### `maj-data.py` — import / mise à jour des débits

Pour chaque raccordement `flex_strategy == "DEFAULT"` (les 3 centrales gardées
le sont toutes), on traite **la station de référence**
(`station_vigicrue_reference`) **et toutes les stations amont**
(`stations_vigicrue_amont`, codes séparés par des virgules). Pour chaque station :

- code jamais vu ailleurs → **import** complet `01/01/2021 → aujourd'hui` (eaufrance) ;
- code déjà porté par un autre dossier → **symlink** local vers ce fichier réel (dédup, pas de nouvel appel API) ;
- CSV déjà présent pour ce dossier → **mise à jour** incrémentale : Hub'Eau `observations_tr` si plus
  récent que eaufrance (agrégé en moyenne horaire), sinon eaufrance seule.

Sortie : `centrales/<dossier>/<station>.csv` (référence) ou `amont_<station>.csv`
(stations amont) — stockage local direct (`config.NAS_DATA_ROOT ==
config.CENTRALES_DIR` dans cette version, plus de NAS distant).

```bash
python cron/scripts/maj-data.py                  # import/MAJ (API Hub'Eau/eaufrance réelle)
python cron/scripts/maj-data.py --dossier apas_G1_G4   # test ciblé
python cron/scripts/maj-data.py --end 03/07/2026
```

### `onboarding-bv.py` — caractérisation du bassin versant

Pour chaque centrale (regroupées par site physique quand plusieurs dossiers
partagent le même `centrale_uuid`), mesure le bassin versant amont — via
shapefile connu (`config/bv_mapping.yaml`) ou repli par délimitation MNT
depuis le point exutoire — calcule les paramètres de calage (`K_base`,
`exposition`, `kc_unit`), les points météo NWP représentatifs, les
coordonnées Hub'Eau des stations hydrométriques (référence + amont) et le
temps de transit hydraulique par cross-corrélation saisonnière
(amont→référence : débit vs débit ; référence→centrale : débit vs
`power_output` nettoyé, avec repli géométrique par ratio de distances si la
corrélation directe est peu fiable). Écrit `centrales/<dossier>/bv.json`.
**Idempotent** : un `bv.json` déjà présent n'est jamais retraité sans
`--force` — voir `src/previ_r2d2/preprocessing/bv/README.md` pour le détail
du schéma de sortie et de la logique de repli.

**Sélection des points météo NWP (`stations_meteo_nwp`)** — dans cette
version, `select_meteo_points` (`preprocessing/bv/delineation.py`) est
purement géométrique : génère les nœuds de grille NWP (pas 0.1°) dans
l'emprise du bassin versant, garde ceux dont le centre tombe dans le
polygone, puis, s'il y en a plus que `MAX_METEO_POINTS`, réduit par
k-means sur (latitude, longitude, altitude) et retient le nœud le plus
proche de chaque centroïde. La fonction accepte toujours un paramètre
optionnel `historical_points` (préférence pour des points de grille
couverts dès une période de référence antérieure, utile quand la grille
NWP s'est densifiée avec le temps) mais plus aucun appelant ne le
renseigne dans cette version — la préférence historique est donc un
chemin mort en pratique, la sélection observée est toujours la
purement géométrique. **Limite connue (héritée de la production)** :
certaines régions n'étaient pas couvertes par la grille NWP avant une
certaine date ; pour un raccordement dont le bassin versant tombe dans
une telle région, `data_preparation.csv` peut avoir une portion de son
historique débit sans météo en face (gap réel côté fournisseur, rien à
corriger côté code) — cela ne concerne aucune des 3 centrales gardées
dans ce dépôt.

```bash
python cron/scripts/onboarding-bv.py batch                              # toutes les centrales
python cron/scripts/onboarding-bv.py batch --force                      # recalcule même l'existant
python cron/scripts/onboarding-bv.py single --dossier apas_G1_G4        # test ciblé
```

### `build-data-preparation.py` — Data_Preparation (débit + météo + amont brut)

Construit/met à jour, pour chaque centrale, un CSV historique
`data_preparation.csv` combinant débit brut + météo NWP brute + amont brut,
alignés par horodatage horaire — aucune feature (lags, gradients,
transit_amont saisonnier) n'est calculée ici, ça reste un sous-projet
ultérieur (entraînement/prédiction du modèle hybride meta).

**Lancement manuel** avant un entraînement (la cadence dépend des dates
d'entraînement, pas d'une fréquence fixe). Stage DVC `data_preparation` (cf.
section Pipeline), avec une vraie dépendance sur `debit`/`bv` (via
marqueurs) — `dvc repro dvc.yaml:data_preparation` régénère aussi ces
dépendances amont au passage. Seules les colonnes débit/amont sont vraiment
rafraîchies à chaque exécution ; les colonnes météo restent figées (plus
d'acquisition FTP, cf. note en tête de fichier) tant qu'un sous-projet API
météo publique n'est pas ajouté.

- Débit : station de référence Hub'Eau (les 3 centrales gardées sont toutes
  en `flex_strategy == "DEFAULT"`).
- Amont : colonne `debit_amont` (un seul) ou `debit_amont_{code}` (plusieurs,
  dédupliqués), nommage repris de `lightgbm_model.py` (Previ_v2).
- Météo : points de `bv.json.stations_meteo_nwp` (déjà calculés par
  `onboarding-bv.py`), matching exact `(latitude, longitude)` contre les
  fichiers NWP bruts -- température/précipitation/niveau0° gardés bruts
  (Kelvin, cumul non diffé), la transformation en feature est hors périmètre.
- Historique confirmé (J-1 et avant) uniquement -- la fenêtre temps
  réel/prévision (arbitrage entre runs météo qui se superposent) est hors
  périmètre, réservée à un sous-projet `predict_future_meta` futur.
- Reprise incrémentale : repart juste après le dernier point déjà écrit pour
  cette centrale (pas de fenêtre fixe) ; `--full-history` force un backfill
  complet depuis le début de chaque source.
- `start` est relevé au premier point réel du débit (si postérieur à la
  fenêtre demandée) avant de lire amont/météo -- le débit est la variable
  cible, une donnée météo sans débit en face n'a aucun intérêt pour
  l'entraînement, et ça évite de scanner des années de fichiers NWP pour rien.

**Lecture sans effet de bord** — `debit_source.py`/`amont_source.py` lisent
via le symlink local déjà posé par l'import réel
(`config.CENTRALES_DIR / dossier / station_store.filename_for(...)`),
jamais via `station_store.resolve_nas_path` (qui écrit `_index.json` et peut
créer un symlink local -- effet de bord inapproprié pour un module qui ne fait
que lire une donnée déjà importée par un autre script).

**Piège rencontré (rétention météo en retard)** — dans l'archive météo NWP
figée reprise de la production, un jour donné peut garder toutes ses
échéances (000-360) au lieu de 000-023 seulement (retard ponctuel de
l'ancien pipeline de rétention, retiré dans cette version, cf. note en tête
de fichier), et ses échéances longues (ex. 024) pointent alors vers des
`flow_date` déjà couvertes par les jours suivants -- doublon d'horodatage
bien réel dans l'archive brute (deux runs différents, pas une corruption).
`nwp_reader.read_points` resample chaque point à l'heure avant de combiner
(moyenne les doublons plutôt que de tenter un arbitrage, hors périmètre pour
l'historique confirmé).

**Piège rencontré (format NWP brut a changé dans le temps)** — les fichiers
2021-2024 ont un header différent de 2025+ (`Latitude`/`Longitude`/`2t`
capitalisés + une colonne `hour_index` en plus, vs `latitude`/`longitude`/`2T`
minuscules côté récent) -- `nwp_reader.parse_nwp_file` normalise la casse des
colonnes avant le rename pour accepter les deux formats.

```bash
python cron/scripts/build-data-preparation.py                            # les 3 centrales
python cron/scripts/build-data-preparation.py --dossier apas_G1_G4       # test ciblé
python cron/scripts/build-data-preparation.py --full-history             # backfill complet
```

### `onboarding-check.py` — validation d'un raccordement avant `bv`

Pour chaque raccordement sans `bv.json` encore (pas onboardé), valide son
`config-raccordement.json` (`preprocessing/onboarding/validation.py::missing_fields`)
et journalise les infos manquantes. **Idempotent** : peut être relancé sans
risque (relancé manuellement tant que la config n'est pas complète — plus de
détection automatique de nouveaux raccordements dans cette version, cf. note
en tête de fichier). **Isolation par raccordement** : un enregistrement cassé
ne bloque jamais la validation des autres (`try/except` par item).

```bash
python cron/scripts/onboarding-check.py   # tous les raccordements pas encore onboardés (pas de --dossier)
```

### `train.py` — entraînement automatisé (2 cadences)

- `run_new_dossiers()` (quotidien) : pour chaque horizon **sans** modèle en
  production, rafraîchit `data_preparation.csv` pour ce dossier puis entraîne
  dès que 12 mois d'historique sont atteints.
- `run_monthly_retrain()` (mensuel) : pour chaque horizon **déjà** en
  production, réentraîne inconditionnellement (l'invocation mensuelle est
  l'échéance) — promeut seulement si le nouveau modèle est meilleur (KGE sur
  le même holdout que le modèle en prod, cf. `model/pipeline/promotion.py`).

```bash
python cron/scripts/train.py --mode new       # stage DVC train_new
python cron/scripts/train.py --mode monthly   # stage DVC train_monthly
python cron/scripts/train.py --dossier apas_G1_G4 --horizon 8 --force   # test manuel ciblé, ignore l'éligibilité
```

Le modèle en PRODUCTION vit dans `models/<dossier>/h<horizon>/`, versionné
DVC + tag git (`<dossier>-h<horizon>-v<N>`, rollback = `git checkout <tag> &&
dvc pull`) — jamais dans `weights/hybrid/` (zone de travail manuelle) ni
`weights/hybrid_candidate/` (candidat en cours d'évaluation, jamais lu par la
prédiction).

> **Attention (tests `slow`)** — `promote_model` fait un vrai `dvc add` +
> `git add` + `git commit` + `git tag` sur le dépôt courant, pas une
> simulation. Les tests d'intégration marqués `slow`
> (`tests/integration/test_train_predict_e2e.py`) appellent le vrai
> `train_one` -> `promote_model` : les lancer pour de vrai (`pytest -m slow`)
> crée un commit + tag réels sur la branche courante à chaque exécution
> (nouvelle version `v2`, `v3`, ... à chaque relance). C'est volontaire
> (démonstration pédagogique du mécanisme de promotion réel), pas un mock à
> corriger — mais soyez-en conscient avant de lancer la suite `slow` sur une
> branche que vous ne voulez pas polluer de commits. Pensez aussi à
> `git push`/`dvc push` après (cf. section « Travail en équipe » en tête de
> fichier) pour partager le nouveau modèle.

### `predict-archive.py` — prédiction + archivage horaire

Pour chaque (dossier, horizon) ayant un modèle en production : archive le
JSON de prévision existant (`postprocessing/archive.py::archive_previous_json`
— l'heure de production, pas l'heure d'archivage, dans le nom de fichier)
vers `./ARCHIVE/<dossier>/<AAAA>/<MM>/<JJ>/` (local, `config.ARCHIVE_ROOT`),
puis prédit la nouvelle heure et écrit `centrales/<dossier>/prevision.json`
(h8) ou fusionne dans `enchere.json` (clés `J2`/`J3`, h48/h72). Isolation par
(dossier, horizon).

`run_prediction`/`load_prediction_window` acceptent un paramètre
`source="live"` (défaut, assemble la fenêtre dynamiquement comme
`build-data-preparation.py`) ou `source="frozen"` (relit directement
`data_preparation.csv` tel quel, sans aucun appel réseau) — utile pour une
prédiction 100% reproductible, par exemple en test.

```bash
python cron/scripts/predict-archive.py   # toutes les centrales avec un modèle en prod (pas de --dossier)
```

### `src/previ_r2d2/model/` — portage du modèle hybride meta

Fonctions pures (pas de classe à état, sauf `BiLSTMHydro` qui est un
`nn.Module` PyTorch — contrainte du framework, pas un choix indépendant).
Convention de placement : logique propre à une seule architecture (LGBM ou
BiLSTM) → `model/architectures/<nom>/` ; logique qui combine plusieurs
architectures (Stacking, prédiction test set, plots) → `model/pipeline/`.

- `model/features/` — feature engineering, port fidèle de `feature()`
  (`lightgbm_model.py` Previ_v2) : `et0.py`, `snow.py`, `meteo_hydro.py`
  (météo/hydrologie), `debit_autoregressif.py` (lags/gradients/récession/
  baseflow du débit cible), `amont.py` (stations amont + transit saisonnier,
  `shift_amont_columns` réutilisée par l'orchestrateur pour les séquences BiLSTM).
- `model/architectures/lightgbm/` — sous-projet LightGBM (flux "meta" actif
  uniquement, pas le flux "v4 legacy") : `metrics.py` (KGE, quantiles,
  poids, post-traitement), `features.py` (`build_features`), `training.py`
  (sélection de features, tuning Optuna, OOF, entraînement final —
  dépendances `lightgbm`/`optuna`), `predict.py` (`predict_lgbm_full` —
  prédiction sur un DataFrame complet), `explain.py` (SHAP beeswarm/bar).
- `model/architectures/bilstm/` — sous-projet BiLSTM : `metrics.py` (KGE
  différentiable + wrapper numpy), `sequences.py` (fenêtres glissantes,
  sélection des colonnes de séquence), `model.py` (classe `BiLSTMHydro` —
  LSTM bidirectionnel + attention + `fit_oof` + `predict` — dépendance
  `torch`, CPU-only), `explain.py` (heatmaps d'attention sur les pics de crue).
- `model/architectures/stacking.py` — meta-learner Stacking/Ridge : fonctions
  pures (pas de classe — l'API classe de Previ_v2 s'est révélée non utilisée
  en prod), combinent les sorties OOF LightGBM/BiLSTM + features
  contextuelles pour le meta-learner final.
- `model/pipeline/` — **orchestration entraînement**, assemble toutes les
  briques ci-dessus (port de `train_meta.py`/`run_one`, Previ_v2) :
  `bv_config.py` (assembleurs `bv.json` → `bv_params`/`transit_amont`),
  `data_loading.py` (chargement/split train-test), `oof_cache.py`
  (cache disque OOF LGB/BiLSTM), `stacking_fit.py` (fit du meta-learner),
  `predict.py` (`predict_test_set` — prédiction test set sans dupliquer
  `StackingHydroModel.predict()`), `metrics.py` (KGE par pas/régime/saison,
  interprétabilité), `artifacts.py` (assemblage + écriture `results.json`/
  `meta_config.json`/CSV), `plots.py` (7 plots : SHAP, attention, comparaison
  test, KGE-by-step, coefficients Ridge, KGE radar), `orchestrator.py`
  (`run_training` — point d'entrée qui appelle tout dans l'ordre).
- **`model/pipeline/predict_orchestrator.py`** — prédiction opérationnelle
  (`run_prediction`, port de `predict_future_meta.py` Previ_v2) : lit le
  modèle depuis `models/<dossier>/h<horizon>/` (production, jamais
  `weights/hybrid/`), débit station + conversion turbine (`hydraulic.py` —
  puissance/chute/rendement).
- **`model/pipeline/{eligibility,promotion}.py`** — éligibilité (12 mois /
  réentraînement mensuel) et promotion versionnée (comparaison KGE candidat
  vs production sur le même holdout, `dvc add`+`git tag`, sûr face à un échec
  partiel). Cf. section `train.py` ci-dessus pour le détail opérationnel.

Voir le skill `previ-r2d2` pour le détail pièce par pièce et le skill
`hybrid-meta-ops` pour l'architecture du modèle hybride.

```bash
.venv/bin/python -m pytest tests/model/ -v   # tests du portage modèle
```

### `run.py` — entraînement/prédiction MANUELS (expés, pas le chemin automatisé)

CLI mince (`src/previ_r2d2/cli.py`) qui appelle `run_training`/`run_prediction`
pour un dossier+horizon donné, ou pour toutes les centrales × 3 horizons
(8h/48h/72h). **Le chemin opérationnel automatisé passe par
`cron/scripts/train.py`/`predict-archive.py`** (cf. sections dédiées
ci-dessus), pas par `run.py` — celui-ci reste réservé aux tests/expés
manuels (entraîne toujours dans `weights/hybrid/`, jamais `models/`, donc
jamais lu par la prédiction opérationnelle sans passer par une promotion
explicite).

```bash
python run.py --train --dossier apas_G1_G4 --horizon 8
python run.py --train --all-dossiers                      # toutes les centrales x h8/h48/h72
python run.py --train --dossier apas_G1_G4 --horizon 8 --force-lgbm --force-lstm  # recalcul complet
python run.py --train --dossier apas_G1_G4 --horizon 8 --meta lgbm --epochs 60
```

Sorties : `weights/hybrid/<dossier>/h<horizon>/` (`results.json`,
`meta_config.json`, poids/caches OOF, `plots/`) et
`outputs/hybrid/<dossier>/h<horizon>/` (CSV + PNG de comparaison test).
Chaque paire (dossier, horizon) échoue indépendamment en mode
`--all-dossiers` (log + continue) — code de sortie `1` si au moins un échec.

## Flux normal (bout en bout, débit → prédiction)

```bash
python cron/scripts/maj-data.py                             # 1. importe/complète les débits (Hub'Eau)
python cron/scripts/onboarding-check.py                     # 2. valide les raccordements pas encore onboardés
python cron/scripts/onboarding-bv.py batch                  # 3. caractérise le BV + stations/transit (no-op si déjà fait)
python cron/scripts/train.py --mode new                     # 4. 1er entraînement des nouvelles centrales
python cron/scripts/train.py --mode monthly                 # 5. réentraînement mensuel (promotion conditionnelle)
python cron/scripts/predict-archive.py                      # 6. archive + prédit la nouvelle heure
```

## Plateforme MLOps (projet de cours)

Le cœur ML ci-dessus est enveloppé d'une couche MLOps en 8 phases (branches
`feat/mlops-phase-N`, une PR par phase). Tout via `make` + `docker compose` :

| | Commande | UI |
|---|---|---|
| Stack MLflow | `make up` | :5000 · MinIO :9001 |
| API de service | `make api` | :8000/docs |
| Orchestration Prefect | `make prefect` | :4200 |
| Monitoring | `make monitor` | Grafana :3000 · Prometheus :9090 |
| Reverse-proxy | `make proxy` | :8080 |
| Démo bout-en-bout | `make demo` | — |
| Chart Helm (API k8s) | `make helm-lint` / `make k8s-deploy` | — |

Détail : **[`ARCHITECTURE.md`](ARCHITECTURE.md)**, **[`MLOPS.md`](MLOPS.md)**,
[`docs/mlops/`](docs/mlops/).

## Références

- Hub'Eau hydrométrie : https://hubeau.eaufrance.fr/page/api-hydrometrie
