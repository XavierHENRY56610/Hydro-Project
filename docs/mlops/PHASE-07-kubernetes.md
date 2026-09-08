# Phase 07 — Déploiement Kubernetes + Nginx

## Périmètre

À l'issue des phases 00–06 la stack complète tourne en `docker compose`
(contrainte fondatrice). La phase 07 **ré-orchestre l'API** — le composant qui
a vocation à encaisser du trafic — avec Kubernetes + un chart Helm, et ajoute
**Nginx** en point d'entrée unique.

Le pipeline / l'entraînement / MLflow / Prometheus ne sont **pas** portés en
k8s ici (documenté comme extension) : ils restent en compose, l'API k8s
pointe dessus via `MLFLOW_TRACKING_URI`.

## Nginx (reverse-proxy)

`infrastructure/nginx/nginx.conf` + service `nginx` dans `docker-compose.yml` :
point d'entrée unique de la stack compose.

| Chemin | Backend |
|---|---|
| `/` | `api:8000` |
| `/mlflow/` | `mlflow:5000` |
| `/grafana/` | `grafana:3000` |
| `/prefect/` | `prefect-server:4200` |
| `/nginx-health` | `200 ok` (sonde) |

Propage `X-Request-ID`, `X-Forwarded-*`. En k8s, ce rôle est tenu par
l'**Ingress nginx** (`templates/ingress.yaml`).

```bash
make proxy                     # nginx + api + mlflow
curl http://localhost:8080/health
```

## Chart Helm (`infrastructure/helm/previ-r2d2/`)

| Template | Ressource |
|---|---|
| `deployment.yaml` | `Deployment` — 2 réplicas (ou piloté par le HPA), probes `/health`, `envFrom` secret, volume `models` |
| `service.yaml` | `Service` ClusterIP |
| `ingress.yaml` | `Ingress` classe `nginx`, hôte `previ-r2d2.local`, TLS optionnel |
| `hpa.yaml` | `HorizontalPodAutoscaler` v2 — 2→6 pods, cible CPU 70 % |
| `pvc.yaml` | `PersistentVolumeClaim` — cache des artefacts modèle / repli `models/` |
| `NOTES.txt` | instructions post-install |

Config par `values.yaml` : image, réplicas, ressources, env
(`MLFLOW_TRACKING_URI`, `API_MODEL_SOURCE`), ingress, autoscaling, persistance.
Les secrets viennent d'un `Secret` externe (`existingSecret`,
cf. `secret.example.yaml`) — jamais dans les values.

## Déploiement

```bash
# validation (helm via conteneur, pas d'install locale nécessaire)
make helm-lint
make helm-template          # revue des manifests rendus

# cluster : Docker Desktop Kubernetes (à cocher dans les réglages) ou k3d
kubectl create secret generic previ-r2d2-secrets \
  --from-literal=API_JWT_SECRET="$(openssl rand -hex 32)" \
  --from-literal=AWS_ACCESS_KEY_ID=minioadmin \
  --from-literal=AWS_SECRET_ACCESS_KEY=minioadmin \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=http://minio:9000

make k8s-deploy             # helm upgrade --install previ ./... --wait
kubectl get pods,svc,ingress,hpa -l app.kubernetes.io/instance=previ
kubectl scale deploy/previ-previ-r2d2 --replicas=3   # (si HPA désactivé)
make k8s-delete
```

Pour un cluster sans Ingress controller :
`helm template previ ./infrastructure/helm/previ-r2d2 | kubectl apply -f -`
puis `kubectl port-forward svc/previ-previ-r2d2 8000:80`.

## Validé

- `helm lint` : 0 échec ; `helm template` rend 5 ressources, YAML valide.
- `tests/infra/test_helm_and_proxy.py` — cohérence Chart/values/templates +
  upstreams nginx ↔ services compose (6 tests).
- Nginx : `make proxy` puis `curl :8080/health` → réponse de l'API.

> Déploiement k8s réel : nécessite un cluster (Docker Desktop Kubernetes ou
> k3d). Non exécuté dans le run de construction — commandes ci-dessus prêtes.

## Extensions documentées

- Porter MLflow (`StatefulSet` PostgreSQL + `PVC`) et Prometheus/Grafana
  (`kube-prometheus-stack`) en k8s.
- Pipeline : `CronJob` k8s par cadence, ou worker Prefect déployé dans le
  cluster (décision de cadrage, cf. phase 05).
- `cert-manager` pour le TLS automatique de l'Ingress.
