"""Configuration centrale du projet.

Chemins internes (racine, centrales/, models/, ...) : dérivés de
l'emplacement de ce fichier, jamais configurables.

Secrets / chemins externes (MNT, météo NWP, puissance, DagsHub, MLflow) :
lus par `Settings` (pydantic-settings), par ordre de priorité :
  1. `secret_config.py` s'il existe (fichier local, NON versionné) ;
  2. variables d'environnement / fichier `.env` (injectées par docker-compose) ;
  3. valeurs par défaut ("" -> dégradation gracieuse).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Chemins internes (non configurables) ---------------------------------
# Racine du projet : src/previ_r2d2/common/config.py -> src/previ_r2d2/ -> src/ -> racine.
ROOT = Path(__file__).resolve().parents[3]
CENTRALES_DIR = ROOT / "centrales"
REFERENCE_DIR = CENTRALES_DIR / "REFERENCE"
# Un seul niveau de stockage (plus de NAS distant) : alias de CENTRALES_DIR.
NAS_DATA_ROOT = CENTRALES_DIR
# models/<dossier>/h<horizon>/ = modèle en production (versionné DVC + tag git).
MODELS_DIR = ROOT / "models"
# Archive locale des prévisions horaires (gitignorée).
ARCHIVE_ROOT = ROOT / "ARCHIVE"

# --- Module de secrets local (optionnel, jamais commité) ------------------
try:
    from . import secret_config as _secret  # type: ignore
except ImportError:
    _secret = None


class Settings(BaseSettings):
    """Secrets et chemins externes. secret_config.py > env / .env > défaut."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Chemins externes historiques ("" => dégradation gracieuse).
    previ_mnt: str = ""
    previ_nas_meteo: str = ""
    previ_puissance_source_root: str = ""

    # DagsHub : git + remote DVC (token perso, configuré dans le conteneur).
    dagshub_user: str = ""
    dagshub_token: str = ""

    # MLflow (phase 01) — vide en phase 00.
    mlflow_tracking_uri: str = ""

    # API de service (phase 03).
    # api_auth_required=True => tous les endpoints métier exigent un Bearer JWT
    # signé avec api_jwt_secret (HS256). Vide/False => API ouverte (démo, dev).
    api_jwt_secret: str = ""
    api_auth_required: bool = False
    # models:/previ-r2d2-<dossier>-h<horizon>@production chargé depuis le
    # registry si dispo ; sinon fallback sur models/<dossier>/h<horizon>/ (DVC).
    api_model_source: str = "auto"  # auto | registry | local

    def with_secret_overrides(self) -> Settings:
        """secret_config.py a la priorité absolue (compat historique)."""
        if _secret is None:
            return self
        updates = {
            field: getattr(_secret, field.upper())
            for field in type(self).model_fields
            if hasattr(_secret, field.upper())
        }
        return self.model_copy(update=updates) if updates else self


settings = Settings().with_secret_overrides()


def _get(name: str, default: str = "") -> str:
    """Compat : valeur d'un réglage par son nom d'env (ex. "PREVI_MNT")."""
    field = name.lower()
    if field in type(settings).model_fields:
        return getattr(settings, field) or default
    if _secret is not None and hasattr(_secret, name):
        return getattr(_secret, name)
    return os.environ.get(name, default)


# --- Chemins externes exposés (Path, comme avant) ------------------------
PUISSANCE_SOURCE_ROOT = Path(settings.previ_puissance_source_root)
NAS_METEO = Path(settings.previ_nas_meteo)
PREVI_MNT = Path(settings.previ_mnt)
