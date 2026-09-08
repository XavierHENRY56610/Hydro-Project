"""Le script de démo (phase 08) doit s'importer proprement et exposer `main`.

Le déroulé complet (entraînement + promotion réels) n'est pas exécuté ici —
ses briques sont couvertes par tests/model/ et tests/cron/test_train.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "demo_script", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "demo.py"
)
demo_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_script)


def test_demo_module_shape():
    assert callable(demo_script.main)
    assert demo_script.DOSSIER == "demo_centrale"
    assert demo_script.HORIZON in (8, 48, 72)
    assert "exutoire" in demo_script.BV_JSON
