# Workflows CI — à activer manuellement

Les fichiers `ci.yml` / `release.yml` de ce dossier sont les workflows
GitHub Actions de la phase 04. Ils sont **parqués ici** parce que le PAT
utilisé pour pousser cette branche n'a pas le scope `workflow` (GitHub
refuse alors de créer/modifier `.github/workflows/*`).

## Pour les activer

Avec un token ayant le scope `workflow` (ou depuis l'UI GitHub) :

```bash
mkdir -p .github/workflows
git mv infrastructure/ci/ci.yml      .github/workflows/ci.yml
git mv infrastructure/ci/release.yml .github/workflows/release.yml
git rm infrastructure/ci/README.md
git commit -m "ci: activer les workflows GitHub Actions (phase 04)"
git push
```

Ou, plus simple : copier-coller le contenu des deux fichiers via
*Add file → Create new file* sur github.com, chemin
`.github/workflows/ci.yml` et `.github/workflows/release.yml`.

## Ensuite

*Settings → Branches* → règle de protection sur `import-projet` : exiger le
job **`lint-test`** vert avant merge.
