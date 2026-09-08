"""Authentification Bearer JWT optionnelle (phase 03).

Désactivée par défaut (`api_auth_required=False`) : l'API est ouverte, adaptée
à la démo et au dev. Activée, chaque endpoint métier exige un JWT HS256 valide
signé avec `api_jwt_secret` ; `GET /health` et `GET /metrics` restent publics
(sondes k8s / scrape Prometheus).
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from previ_r2d2.common import config

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def auth_enabled() -> bool:
    return bool(config.settings.api_auth_required and config.settings.api_jwt_secret)


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    """Dépendance FastAPI : no-op si l'auth est désactivée, sinon valide le JWT."""
    if not auth_enabled():
        return

    if credentials is None or not credentials.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "jeton Bearer requis")

    import jwt

    try:
        jwt.decode(credentials.credentials, config.settings.api_jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        logger.warning("rejet JWT : %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "jeton invalide ou expiré") from exc
