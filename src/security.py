"""
security.py — Garde-fou de confidentialité des données RH.

Empêche le chatbot de divulguer des données sensibles (matricule, salaire,
rémunération…) aux utilisateurs non autorisés.

Politique appliquée :
    - role == "superadmin"  → accès complet (RH/administration).
    - role == "employee"    → données sensibles bloquées.

Deux niveaux de défense :
    1. is_sensitive_question() — intercepte la QUESTION avant tout traitement
       (rapide, économise une recherche inutile).
    2. redact_sensitive_columns() / scrub_text() — filtre la RÉPONSE en sortie
       (filet de sécurité si une donnée sensible passe malgré tout).
"""

import re
from typing import Iterable, List, Optional

# Rôles autorisés à voir TOUTES les données.
PRIVILEGED_ROLES = {"superadmin"}

# ── Catégories de données sensibles ─────────────────────────────────────────
# Chaque entrée : (libellé, motifs regex). Les motifs ciblent à la fois les
# intitulés de colonnes Excel et les mots-clés des questions en langage naturel.
# Tolérant aux accents (e/é), au pluriel et aux variantes courantes.

SENSITIVE_CATEGORIES = {
    "matricule": [
        r"matricul",          # matricule, matricules
        r"\bn[°o]?\s*ss\b",   # n° SS, NSS
        r"s[ée]curit[ée]\s+sociale",
        r"num[ée]ro\s+de\s+s[ée]curit[ée]",
        r"\bnss\b",
        r"identifiant\s+(?:employ|salari|agent|personnel)",
        r"\bn[°o]?\s*(?:de\s+)?compte\b",
        r"\bcompte\s+bancaire\b",
        r"\briberib\b|\brib\b",
    ],
    "salaire": [
        r"\bmon\s+salair",                        # mon salaire
        r"\bson\s+salair",                        # son salaire
        r"\bsalair\w*\s+de\s+(?:m\.|mr?\.?\s)?\w+",  # salaire de M. X
        r"\bma\s+r[ée]mun[ée]ra",                # ma rémunération
        r"\bsa\s+r[ée]mun[ée]ra",                # sa rémunération
        r"bulletin\s+de\s+(?:paie|salaire)",
        r"fiche\s+de\s+paie",
        r"\bgaranti\s+mensuel",
        r"\bnet\s+[àa]\s+payer\b",
        r"\bsalaire\s+(?:net|brut)\b",           # salaire net/brut (personnel)
    ],
}

# Pré-compilation des motifs (performance + lisibilité).
_COMPILED = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in SENSITIVE_CATEGORIES.items()
}

# Message renvoyé à l'utilisateur lorsqu'une demande est refusée.
REFUSAL_MESSAGE = (
    "Pour des raisons de confidentialité, je ne peux pas communiquer ce type "
    "d'information ({categories}). Ces données personnelles sont réservées au "
    "service des Ressources Humaines. Merci de vous adresser directement à la DRH."
)

_CATEGORY_LABELS = {
    "matricule": "matricule, numéro de sécurité sociale, coordonnées bancaires",
    "salaire":   "salaire, rémunération, éléments de paie",
}


def is_privileged(role: Optional[str]) -> bool:
    """True si le rôle a accès à l'ensemble des données (aucun filtrage)."""
    return (role or "").lower().strip() in PRIVILEGED_ROLES


def detect_categories(text: str) -> List[str]:
    """Retourne la liste des catégories sensibles présentes dans `text`."""
    if not text:
        return []
    found = []
    for category, regexes in _COMPILED.items():
        if any(rx.search(text) for rx in regexes):
            found.append(category)
    return found


def is_sensitive_question(question: str, role: Optional[str]) -> Optional[str]:
    """Vérifie une question AVANT traitement.

    Retourne le message de refus si la question d'un utilisateur non privilégié
    porte sur des données sensibles, sinon None (la question peut être traitée).
    """
    if is_privileged(role):
        return None
    categories = detect_categories(question)
    if not categories:
        return None
    labels = "; ".join(_CATEGORY_LABELS[c] for c in categories)
    return REFUSAL_MESSAGE.format(categories=labels)


def is_sensitive_column(column_name: str) -> bool:
    """True si l'intitulé d'une colonne Excel correspond à une donnée sensible."""
    return bool(detect_categories(column_name))


def filter_sensitive_columns(columns: Iterable[str], role: Optional[str]) -> List[str]:
    """Retire les colonnes sensibles d'une liste, sauf pour les rôles privilégiés."""
    if is_privileged(role):
        return list(columns)
    return [c for c in columns if not is_sensitive_column(c)]


def redact_row(row: dict, role: Optional[str]) -> dict:
    """Retire les colonnes sensibles d'une ligne Excel (dict colonne→valeur).

    Utilisé sur les `raw_row` issus de DuckDB avant tout formatage de réponse,
    pour que matricule/salaire ne se retrouvent jamais dans la sortie d'un
    utilisateur non privilégié. Les rôles privilégiés reçoivent la ligne intacte.
    """
    if not row or is_privileged(role):
        return row
    return {k: v for k, v in row.items() if not is_sensitive_column(str(k))}


# Masque les paires "Libellé: valeur" / "Libellé = valeur" / "[Libellé] valeur"
# dont le libellé est sensible — format produit par les nœuds structurés.
_PAIR_RE = re.compile(
    r"""
    (?P<label>[A-Za-zÀ-ÿ ._/'-]{2,40}?)   # libellé de colonne
    \s*(?P<sep>[:=])\s*                    # séparateur : ou =
    (?P<value>[^\n;|]+)                     # valeur jusqu'au prochain délimiteur
    """,
    re.VERBOSE,
)

REDACTED_PLACEHOLDER = "[confidentiel]"


def scrub_answer(answer: str, role: Optional[str]) -> str:
    """Filet de sécurité (niveau 2) : masque dans le TEXTE de réponse les paires
    « libellé sensible : valeur » qui auraient pu subsister.

    N'altère rien pour les rôles privilégiés. Conçu pour les sorties structurées
    de la forme "Matricule: 12345" → "Matricule: [confidentiel]".
    """
    if not answer or is_privileged(role):
        return answer

    def _mask(m: "re.Match") -> str:
        if is_sensitive_column(m.group("label")):
            trailing = " " if m.group("value").endswith(" ") else ""
            return f"{m.group('label')}{m.group('sep')} {REDACTED_PLACEHOLDER}{trailing}"
        return m.group(0)

    return _PAIR_RE.sub(_mask, answer)
