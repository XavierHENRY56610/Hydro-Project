"""Vérifs statiques du chart Helm (phase 07) et du reverse-proxy nginx.

Pas de `helm` ni de cluster ici (couverts par `make helm-lint` / `make
helm-template` en local) : on valide la cohérence des fichiers — Chart.yaml,
values.yaml, présence des templates attendus, upstreams nginx alignés sur les
services compose.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "infrastructure" / "helm" / "previ-r2d2"


def test_chart_metadata():
    meta = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    assert meta["name"] == "previ-r2d2"
    assert meta["apiVersion"] == "v2"
    assert "version" in meta and "appVersion" in meta


def test_values_expose_expected_keys():
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["image"]["repository"].startswith("ghcr.io/")
    assert values["ingress"]["className"] == "nginx"
    assert values["autoscaling"]["enabled"] is True
    assert values["livenessProbe"]["httpGet"]["path"] == "/health"


def test_expected_templates_present():
    names = {p.name for p in (CHART / "templates").glob("*")}
    assert {"deployment.yaml", "service.yaml", "ingress.yaml", "hpa.yaml",
            "pvc.yaml", "_helpers.tpl", "NOTES.txt"} <= names


def test_templates_use_fullname_helper():
    for tpl in ("deployment.yaml", "service.yaml", "ingress.yaml", "hpa.yaml", "pvc.yaml"):
        text = (CHART / "templates" / tpl).read_text(encoding="utf-8")
        assert 'include "previ-r2d2.fullname"' in text or 'include "previ-r2d2.labels"' in text


def test_nginx_upstreams_match_compose_services():
    nginx_conf = (ROOT / "infrastructure" / "nginx" / "nginx.conf").read_text(encoding="utf-8")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = set(compose["services"])
    for upstream_host in ("api:8000", "mlflow:5000", "grafana:3000", "prefect-server:4200"):
        name = upstream_host.split(":")[0]
        assert upstream_host in nginx_conf
        assert name in services
    assert "location = /nginx-health" in nginx_conf
    assert "resolver 127.0.0.11" in nginx_conf  # résolution DNS paresseuse


def test_compose_nginx_service_fronts_api():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    nginx = compose["services"]["nginx"]
    assert "api" in nginx["depends_on"]
    assert any("80" in str(p) for p in nginx["ports"])
