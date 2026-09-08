"""API de service previ-R2-D2 (phase 03).

`api.py` expose le modèle hybride en HTTP (REST + JSON) : prévision à la
demande, lecture des dernières prévisions archivées, inventaire des modèles
en production, `/health` et `/metrics` (Prometheus).

Cadrage production repris des cours DataScientest (FastAPI, LLM en
Production) : logs JSON structurés, en-tête `X-Request-ID`, erreurs sans
stack trace côté client, auth Bearer JWT optionnelle.
"""
