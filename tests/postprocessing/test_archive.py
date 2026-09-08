from __future__ import annotations

import pandas as pd

from previ_r2d2.postprocessing.archive import archive_previous_json


def test_archive_previous_json_copies_existing_files(tmp_path):
    centrales_dir = tmp_path / "centrales"
    archive_root = tmp_path / "ARCHIVE"
    dossier_dir = centrales_dir / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "prevision.json").write_text('{"a": 1}', encoding="utf-8")
    (dossier_dir / "enchere.json").write_text('{"b": 2}', encoding="utf-8")

    # now = heure de l'archivage (10h). Le prevision.json déjà présent a été
    # produit lors du run précédent (9h) : le nom archivé doit porter l'heure
    # de production (now - 1h), pas l'heure d'archivage.
    now = pd.Timestamp("2026-07-16 14:00:00")
    archived = archive_previous_json("apas_G1_G4", centrales_dir, archive_root, now)

    assert len(archived) == 2
    dest_dir = archive_root / "apas_G1_G4" / "2026" / "07" / "16"
    assert (dest_dir / "prevision_20260716_13h.json").read_text(encoding="utf-8") == '{"a": 1}'
    assert (dest_dir / "enchere_20260716_13h.json").read_text(encoding="utf-8") == '{"b": 2}'
    # Le fichier source n'est pas supprimé (copie, pas déplacement) -- la
    # prédiction suivante l'écrasera elle-même avec la nouvelle heure.
    assert (dossier_dir / "prevision.json").exists()


def test_archive_previous_json_uses_previous_day_at_midnight_rollover(tmp_path):
    centrales_dir = tmp_path / "centrales"
    archive_root = tmp_path / "ARCHIVE"
    dossier_dir = centrales_dir / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "prevision.json").write_text('{"a": 1}', encoding="utf-8")

    # Archivage à 00h le 16 : le fichier existant a été produit à 23h le 15.
    now = pd.Timestamp("2026-07-16 00:00:00")
    archived = archive_previous_json("apas_G1_G4", centrales_dir, archive_root, now)

    dest_dir = archive_root / "apas_G1_G4" / "2026" / "07" / "15"
    assert (dest_dir / "prevision_20260715_23h.json").read_text(encoding="utf-8") == '{"a": 1}'
    assert archived == [dest_dir / "prevision_20260715_23h.json"]


def test_archive_previous_json_returns_empty_list_when_no_json_yet(tmp_path):
    centrales_dir = tmp_path / "centrales"
    (centrales_dir / "apas_G1_G4").mkdir(parents=True)
    archive_root = tmp_path / "ARCHIVE"

    archived = archive_previous_json("apas_G1_G4", centrales_dir, archive_root, pd.Timestamp.now())

    assert archived == []
