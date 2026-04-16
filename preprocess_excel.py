#!/usr/bin/env python3
"""
Script de prétraitement : Convertit les Excel RH en documents Markdown lisibles par le RAG.
Génère un fichier consolidé par catégorie (au lieu d'un fichier par ligne).
"""

from pathlib import Path
import pandas as pd

# Chemins
RAW_DIR = Path("documents/raw")
STRUCTURED_DIR = Path("documents/structured")
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)


def clean_name(name: str) -> str:
    """Nettoie les noms pour créer des noms de fichiers corrects"""
    if not isinstance(name, str):
        name = str(name)
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip().replace(" ", "_")


# ====================== DIRECTION ======================
print("Conversion des Directions...")

df_dir = pd.read_excel(RAW_DIR / "DIRECTION.xlsx")
sections = []
for _, row in df_dir.iterrows():
    sections.append(f"""## Direction : {row.get('SHORT_LIBELLE_DIRECTION', '')}

- **Code Affectation** : {row.get('AFFECTATION', '')}
- **Chantier** : {row.get('CHANTIER', '')}
- **Responsable** : {row.get('NOM', '')} {row.get('PRENOM', '')}
- **Fonction** : {row.get('FONCTION', '')}
- **Observation** : {row.get('OBSERVATION', '')}""")

content = f"# Directions\n\n" + "\n\n---\n\n".join(sections) + "\n"
(STRUCTURED_DIR / "directions.md").write_text(content, encoding="utf-8")
print(f"  ✓ Créé : directions.md ({len(sections)} entrées)")

# ====================== DEPARTEMENT ======================
print("\nConversion des Départements...")

df_dept = pd.read_excel(RAW_DIR / "DEPARTEMENT.xlsx")
sections = []
for _, row in df_dept.iterrows():
    sections.append(f"""## Département : {row.get('CHANTIER', '')}

- **Code** : {row.get('AFFECTATION', '')}
- **Direction rattachée** : {row.get('SHORT_LIBELLE_DIRECTION', '')}
- **Responsable** : {row.get('NOM', '')} {row.get('PRENOM', '')}
- **Fonction** : {row.get('FONCTION', '')}
- **Observation** : {row.get('OBSERVATION', '')}""")

content = f"# Départements\n\n" + "\n\n---\n\n".join(sections) + "\n"
(STRUCTURED_DIR / "departements.md").write_text(content, encoding="utf-8")
print(f"  ✓ Créé : departements.md ({len(sections)} entrées)")

# ====================== SERVICE ======================
print("\nConversion des Services...")

df_serv = pd.read_excel(RAW_DIR / "SERVICE.xlsx")
sections = []
for _, row in df_serv.iterrows():
    sections.append(f"""## Service : {row.get('CHANTIER', '')}

- **Code** : {row.get('AFFECTATION', '')}
- **Direction / Département** : {row.get('SHORT_LIBELLE_DIRECTION', '')}
- **Responsable** : {row.get('NOM', '')} {row.get('PRENOM', '')}
- **Fonction** : {row.get('FONCTION', '')}
- **Observation** : {row.get('OBSERVATION', '')}""")

content = f"# Services\n\n" + "\n\n---\n\n".join(sections) + "\n"
(STRUCTURED_DIR / "services.md").write_text(content, encoding="utf-8")
print(f"  ✓ Créé : services.md ({len(sections)} entrées)")

# ====================== POSTE ======================
print("\nConversion des Postes...")

df_poste = pd.read_excel(RAW_DIR / "POSTE.xlsx")
sections = []
for _, row in df_poste.iterrows():
    sections.append(f"""## Poste : {row.get('LIBELLE_POSTE', '')}

- **ID Poste** : {row.get('ID', '')}
- **Libellé de base** : {row.get('LIBELLE_POSTE_BASE', '')}
- **Filière** : {row.get('LIBELLE_FILIERE', '')}
- **Sous-filière** : {row.get('LIBELLE_SOUS_FILIERE', '')}
- **Activité** : {row.get('LIBELLE_ACTIVITE', '')}
- **Catégorie** : {row.get('CATEGORIE', '')}""")

content = f"# Postes\n\n" + "\n\n---\n\n".join(sections) + "\n"
(STRUCTURED_DIR / "postes.md").write_text(content, encoding="utf-8")
print(f"  ✓ Créé : postes.md ({len(sections)} entrées)")

print(f"\n✅ Conversion terminée !")
print(f"   → 4 fichiers Markdown consolidés dans documents/structured/")
print("\nMaintenant lance l'ingestion :")
print("   python ingest.py --reset")
