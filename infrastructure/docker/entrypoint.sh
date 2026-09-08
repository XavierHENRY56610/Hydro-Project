#!/bin/sh
# Configure le token DagsHub pour DVC si présent dans l'environnement
# (injecté par docker-compose depuis .env). .dvc/config.local est gitignoré.
set -e

if [ -n "$DAGSHUB_TOKEN" ] && [ -f .dvc/config ]; then
  dvc remote modify dagshub --local user "${DAGSHUB_USER:-token}" >/dev/null 2>&1 || true
  dvc remote modify dagshub --local password "$DAGSHUB_TOKEN" >/dev/null 2>&1 || true
fi

# git a besoin d'une identité pour les commits de promotion (promotion.py).
git config --global --get user.email >/dev/null 2>&1 || \
  git config --global user.email "${GIT_AUTHOR_EMAIL:-previ-r2d2@localhost}"
git config --global --get user.name >/dev/null 2>&1 || \
  git config --global user.name "${GIT_AUTHOR_NAME:-previ-r2d2}"
git config --global --get safe.directory >/dev/null 2>&1 || \
  git config --global --add safe.directory /app

exec "$@"
