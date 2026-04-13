#!/usr/bin/env python3
"""
preprocess_excel.py v2
Génère 3 types de fichiers .md :
  1. Fiches individuelles par personne  → répond aux questions matricule/fonction
  2. Fichiers par Direction (enrichis)  → répond aux questions hiérarchiques
  3. Fichier liste_complete_directions  → répond aux questions de liste globale
  4. Fichiers Postes par Activité
"""
from pathlib import Path
import pandas as pd

RAW_DIR        = Path("documents/raw")
STRUCTURED_DIR = Path("documents/structured")
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

def val(v) -> str:
    if v is None or (isinstance(v, float) and str(v) == 'nan'):
        return 'Non renseigné'
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()

def clean_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return ''.join(c if c.isalnum() or c in ' _-' else '_' for c in name).strip().replace(' ', '_')

def save_md(filename: str, content: str):
    path = STRUCTURED_DIR / filename
    path.write_text(content.strip() + '\n', encoding='utf-8')

# ── Chargement ────────────────────────────────────────────────────────────────
df_dir   = pd.read_excel(RAW_DIR / 'DIRECTION.xlsx')
df_dept  = pd.read_excel(RAW_DIR / 'DEPARTEMENT.xlsx')
df_serv  = pd.read_excel(RAW_DIR / 'SERVICE.xlsx')
df_poste = pd.read_excel(RAW_DIR / 'POSTE.xlsx')

# ── 1. Fiches individuelles par personne ──────────────────────────────────────
print('Génération des fiches individuelles...')
nb = 0
for df, entity_type in [(df_dir, 'Direction'), (df_dept, 'Département'), (df_serv, 'Service')]:
    for _, r in df.iterrows():
        nom = val(r['NOM'])
        if nom == 'Non renseigné':
            continue
        prenom   = val(r['PRENOM'])
        mat      = val(r['MATRICULE'])
        func     = val(r['FONCTION'])
        code     = val(r['SHORT_LIBELLE_DIRECTION'])
        chantier = val(r['CHANTIER']).rstrip('.')
        aff      = val(r['AFFECTATION'])

        lines = [
            f"# Fiche personne : {nom} {prenom}",
            "",
            f"**Nom** : {nom}",
            f"**Prénom** : {prenom}",
            f"**Nom complet** : {nom} {prenom}",
            f"**Matricule** : {mat}",
            f"**Fonction** : {func}",
            f"**Type de structure** : {entity_type}",
            f"**Structure** : {chantier}",
            f"**Direction** : {code}",
            f"**Code affectation** : {aff}",
            "",
            "## Informations complémentaires",
            f"{nom} {prenom} occupe le poste de {func} au sein de la {entity_type} {chantier}, rattachée à la direction {code}.",
            f"Son matricule est {mat} et son code d'affectation est {aff}.",
        ]
        fname = f"personne_{clean_name(nom)}_{clean_name(prenom)}_{val(r['ID'])}.md"
        save_md(fname, '\n'.join(lines))
        nb += 1
print(f"  ✓ {nb} fiches individuelles")

# ── 2. Fichier liste complète des directions (1 document compact) ─────────────
print('Génération liste_complete_directions.md...')
lines = [
    "# Liste complète des Directions Sonatrach",
    "",
    f"Sonatrach compte {len(df_dir)} directions.",
    "",
]
for _, r in df_dir.iterrows():
    resp = f"{val(r['NOM'])} {val(r['PRENOM'])}"
    lines += [
        f"## {val(r['SHORT_LIBELLE_DIRECTION'])} — {val(r['CHANTIER']).rstrip('.')}",
        f"- Code : {val(r['AFFECTATION'])}",
        f"- Directeur : {resp} | Matricule : {val(r['MATRICULE'])}",
        f"- Fonction : {val(r['FONCTION'])}",
        "",
    ]
save_md('liste_complete_directions.md', '\n'.join(lines))
print("  ✓ liste_complete_directions.md")

# ── 3. Fichiers par Direction (avec départements + services + matricules) ──────
print('Génération des fichiers par Direction...')
for _, dir_row in df_dir.iterrows():
    dir_code = val(dir_row['SHORT_LIBELLE_DIRECTION'])
    dir_name = val(dir_row['CHANTIER']).rstrip('.')
    dir_resp = f"{val(dir_row['NOM'])} {val(dir_row['PRENOM'])}"
    dir_mat  = val(dir_row['MATRICULE'])

    lines = [
        f"# Direction {dir_code} — {dir_name}",
        "",
        f"**Nom complet** : {dir_name}",
        f"**Code** : {dir_code} | **Code affectation** : {val(dir_row['AFFECTATION'])}",
        f"**Directeur** : {dir_resp} | **Matricule** : {dir_mat}",
        f"**Fonction** : {val(dir_row['FONCTION'])}",
        "",
    ]

    depts = df_dept[df_dept['SHORT_LIBELLE_DIRECTION'] == dir_code]
    if not depts.empty:
        lines += ["## Départements et Services", ""]
        for _, d in depts.iterrows():
            d_resp = f"{val(d['NOM'])} {val(d['PRENOM'])}"
            lines += [
                f"### {val(d['CHANTIER']).rstrip('.')}",
                f"- Code : {val(d['AFFECTATION'])} | Responsable : {d_resp} | Matricule : {val(d['MATRICULE'])}",
                f"- Fonction : {val(d['FONCTION'])}",
                "",
            ]
            for _, s in df_serv[df_serv['AFFECT_PAR'] == d['ID']].iterrows():
                s_resp = f"{val(s['NOM'])} {val(s['PRENOM'])}"
                lines.append(
                    f"  - {val(s['CHANTIER']).rstrip('.')} | "
                    f"Code: {val(s['AFFECTATION'])} | "
                    f"{s_resp} | Matricule: {val(s['MATRICULE'])} | "
                    f"{val(s['FONCTION'])}"
                )
            lines.append("")

    save_md(f"direction_{clean_name(dir_code)}.md", '\n'.join(lines))
print(f"  ✓ {len(df_dir)} fichiers direction_*.md")

# ── 4. Postes par Activité ────────────────────────────────────────────────────
print('Génération des fichiers Postes...')
nb_act = 0
for activite, group in df_poste.groupby('LIBELLE_ACTIVITE'):
    lines = [
        f"# Postes — Activité : {activite}",
        f"Nombre de postes : {len(group)}",
        "",
        "## Liste des postes",
        "",
    ]
    for _, p in group.iterrows():
        lines += [
            f"### {val(p['LIBELLE_POSTE'])}",
            f"- ID : {val(p['ID'])} | Filière : {val(p['LIBELLE_FILIERE'])} | Catégorie : {val(p['CATEGORIE'])}",
            "",
        ]
    save_md(f"postes_{clean_name(str(activite))[:60]}.md", '\n'.join(lines))
    nb_act += 1
print(f"  ✓ {nb_act} fichiers postes_*.md")

total = len(list(STRUCTURED_DIR.glob('*.md')))
print(f"\n✅ {total} fichiers Markdown créés dans documents/structured/")
print("Lance maintenant : python ingest.py --reset")
