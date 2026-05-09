"""Routage intelligent des requêtes — 100% dynamique.

Aucun nom de table, colonne ou valeur n'est codé en dur.
Tout est découvert automatiquement depuis les fichiers au démarrage.

Architecture :
  1. SchemaDiscovery  : scanne docs_dir → colonnes + échantillons + label_column
  2. SchemaAnalyzer   : analyse le contenu réel pour détecter les "rôles" des colonnes
                        (entity_col, filter_col, person_name_cols, role_col…)
  3. IntentRouter     : construit le prompt LLM entièrement depuis le schéma réel
                        → zéro exemple figé, adaptatif à tout nouveau fichier Excel
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm import HFClient
from ..ingestion.loader import _detect_header_row

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

class SchemaDiscovery:
    """Scanne le dossier docs et extrait colonnes + échantillons pour chaque fichier."""

    EXCEL_EXTS = {".xlsx", ".xls"}
    DOC_EXTS   = {".docx", ".doc", ".pdf", ".txt", ".md", ".html", ".htm"}

    def __init__(self, docs_dir: str, max_samples: int = 5, max_sample_len: int = 50):
        self.docs_dir       = docs_dir
        self.max_samples    = max_samples
        self.max_sample_len = max_sample_len

    def scan(self) -> Dict[str, dict]:
        schema: Dict[str, dict] = {}
        if not Path(self.docs_dir).exists():
            logger.warning("SchemaDiscovery: docs_dir introuvable: %s", self.docs_dir)
            return schema
        for file in sorted(Path(self.docs_dir).rglob("*")):
            if not file.is_file():
                continue
            ext  = file.suffix.lower()
            stem = self._normalize_stem(file.name)
            if ext in self.EXCEL_EXTS:
                entry = self._scan_excel(file)
                if entry:
                    schema[stem] = entry
            elif ext in self.DOC_EXTS:
                schema[stem] = {
                    "columns": [], "samples": {}, "is_doc": True,
                    "filename": file.name,
                }
        logger.info("SchemaDiscovery: %d sources (%s)",
                    len(schema), ", ".join(schema.keys()))
        return schema

    @staticmethod
    def _normalize_stem(fname: str) -> str:
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        stem = stem.upper().strip()
        stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
        stem = re.sub(r"\s*_\d+\s*$",    "", stem)
        return stem.strip()

    def _scan_excel(self, path: Path) -> Optional[dict]:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
            ws = wb.active
            hrow    = _detect_header_row(ws)
            headers = [str(c.value).strip() if c.value else ""
                       for c in ws[hrow]]
            headers = [h for h in headers if h]

            samples: Dict[str, List[str]] = {h: [] for h in headers}
            scanned = 0
            for row in ws.iter_rows(min_row=hrow + 1, values_only=True):
                scanned += 1
                if scanned > 300:
                    break
                for h, v in zip(headers, row):
                    if v is None or len(samples[h]) >= self.max_samples:
                        continue
                    s = str(v).strip()[:self.max_sample_len]
                    # Ignorer les valeurs qui ressemblent à des en-têtes répétés
                    if s and s.upper() != h.upper() and s not in samples[h]:
                        samples[h].append(s)
            wb.close()
            return {
                "columns": headers,
                "samples": {h: v for h, v in samples.items() if v},
                "is_doc":  False,
                "filename": path.name,
            }
        except Exception as e:
            logger.warning("SchemaDiscovery: échec scan %s: %s", path.name, e)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA ANALYZER — détecte les rôles des colonnes à partir du CONTENU réel
# ══════════════════════════════════════════════════════════════════════════════

class SchemaAnalyzer:
    """Analyse le contenu de chaque table pour identifier automatiquement :
    - entity_col   : colonne principale (noms lisibles, ex: CHANTIER, INTITULE_DE_LA_FORMATION)
    - filter_cols  : colonnes catégorielles utiles pour filtrer (ex: STATUT, SHORT_LIBELLE_DIRECTION)
    - name_cols    : colonnes contenant des noms de personnes (NOM, PRENOM)
    - role_col     : colonne contenant la fonction/rôle (FONCTION, POSTE, GRADE)
    """

    # Mots-clés qui indiquent une colonne de "nom de personne"
    _NAME_HINTS   = {"NOM", "PRENOM", "PRÉNOM", "NAME", "FIRSTNAME", "LASTNAME"}
    # Mots-clés qui indiquent une colonne de rôle/fonction
    _ROLE_HINTS   = {"FONCTION", "POSTE", "GRADE", "TITRE", "ROLE", "POSITION",
                     "QUALIFICATION", "EMPLOI"}
    # Préfixes techniques à exclure de entity_col
    _TECH_PREFIXES = ("ID_", "CD_", "NUM_", "SHORT_", "AFFECT_", "MATRICULE")
    _TECH_EXACT    = {"ID", "CODE", "AFFECTATION", "MATRICULE", "OBSERVATION",
                      "AFFECT_PAR", "N"}

    def analyze(self, table_name: str, info: dict) -> dict:
        """Retourne un dict enrichi avec les rôles détectés."""
        columns = info.get("columns", [])
        samples = info.get("samples", {})

        entity_col  = info.get("label_column") or self._detect_entity_col(
            table_name, columns, samples)
        filter_cols = self._detect_filter_cols(columns, samples, exclude=entity_col)
        name_cols   = self._detect_name_cols(columns, samples)
        role_col    = self._detect_role_col(columns, samples)

        return {
            **info,
            "entity_col":  entity_col,
            "filter_cols": filter_cols,   # {col_name: [val1, val2, …]}
            "name_cols":   name_cols,     # ["NOM", "PRENOM"]
            "role_col":    role_col,      # "FONCTION" ou None
        }

    # ── détection entity ──────────────────────────────────────────────────

    def _detect_entity_col(self, table_name: str, columns: List[str],
                            samples: Dict[str, List[str]]) -> Optional[str]:
        """Colonne avec les noms lisibles principaux — plusieurs heuristiques."""
        kw = table_name.upper()

        # Passe 1 : valeurs qui commencent par le nom de la table
        for col in columns:
            if self._is_technical(col):
                continue
            vals = samples.get(col, [])
            if vals and sum(1 for v in vals if v.upper().startswith(kw)) / len(vals) > 0.5:
                return col

        # Passe 2 : colonnes LIBELLE_*, INTITULE_*, DESIGNATION_*, NOM_*
        for prefix in ("LIBELLE_", "INTITULE_", "DESIGNATION_", "NOM_", "TITRE_"):
            for col in columns:
                if col.upper().startswith(prefix) and not col.upper().startswith("SHORT_"):
                    return col

        # Passe 3 : colonne texte la plus longue (exclut techniques)
        best, best_avg = None, 0
        for col in columns:
            if self._is_technical(col):
                continue
            vals = samples.get(col, [])
            if not vals:
                continue
            avg = sum(len(v) for v in vals) / len(vals)
            pct_num = sum(1 for v in vals
                         if v.replace(".", "").replace(",", "").replace("-", "").isdigit()
                         ) / len(vals)
            if avg > best_avg and pct_num < 0.3:
                best, best_avg = col, avg
        return best

    # ── détection filter_cols ─────────────────────────────────────────────

    def _detect_filter_cols(self, columns: List[str],
                             samples: Dict[str, List[str]],
                             exclude: Optional[str]) -> Dict[str, List[str]]:
        """Colonnes catégorielles : peu de valeurs distinctes, texte court."""
        result: Dict[str, List[str]] = {}
        for col in columns:
            if col == exclude or self._is_name_col(col) or col.upper() in self._TECH_EXACT:
                continue
            vals = samples.get(col, [])
            if not vals:
                continue
            avg_len = sum(len(v) for v in vals) / len(vals)
            pct_num = sum(1 for v in vals
                         if v.replace(".", "").replace(",","").replace("-","").isdigit()
                         ) / len(vals)
            # Catégorielle : valeurs courtes (< 40 chars), pas trop numériques
            if avg_len < 40 and pct_num < 0.3:
                # Dédupliquer les valeurs filtre (enlever doublons insensibles casse)
                clean = list({v.strip() for v in vals if v.strip()})
                if 1 < len(clean) <= 10:  # entre 2 et 10 valeurs distinctes = catégorielle
                    result[col] = clean
        return result

    # ── détection name_cols ───────────────────────────────────────────────

    def _detect_name_cols(self, columns: List[str],
                           samples: Dict[str, List[str]]) -> List[str]:
        """Colonnes qui contiennent des noms de personnes."""
        found = []
        for col in columns:
            cu = col.upper()
            # Par nom de colonne
            if any(hint in cu for hint in self._NAME_HINTS):
                found.append(col)
                continue
            # Par contenu : mots en majuscules courts (noms de famille)
            vals = [v for v in samples.get(col, []) if v]
            if vals:
                pct_upper_word = sum(
                    1 for v in vals
                    if re.match(r"^[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ\s\-]+$", v)
                ) / len(vals)
                if pct_upper_word > 0.6 and not self._is_technical(col):
                    found.append(col)
        return found

    # ── détection role_col ────────────────────────────────────────────────

    def _detect_role_col(self, columns: List[str],
                          samples: Dict[str, List[str]]) -> Optional[str]:
        """Colonne qui contient les fonctions/rôles des personnes."""
        for col in columns:
            cu = col.upper()
            # Par nom de colonne
            if any(hint in cu for hint in self._ROLE_HINTS):
                return col
            # Par contenu : phrases commençant par un verbe de rôle
            vals = samples.get(col, [])
            if vals:
                role_kw = ("CHEF", "DIRECTEUR", "RESPONSABLE", "CHARGE",
                           "INGENIEUR", "TECHNICIEN", "ASSISTANT", "AGENT")
                pct_role = sum(
                    1 for v in vals
                    if any(v.upper().startswith(kw) for kw in role_kw)
                ) / len(vals)
                if pct_role > 0.5:
                    return col
        return None

    # ── helpers ───────────────────────────────────────────────────────────

    def _is_technical(self, col: str) -> bool:
        cu = col.upper()
        return (cu in self._TECH_EXACT or
                any(cu.startswith(p) for p in self._TECH_PREFIXES))

    def _is_name_col(self, col: str) -> bool:
        cu = col.upper()
        return any(hint in cu for hint in self._NAME_HINTS)


# ══════════════════════════════════════════════════════════════════════════════
# INTENT ROUTER — prompt 100% dynamique
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "Tu es un classifieur de requêtes en français. "
    "Tu retournes UNIQUEMENT un objet JSON valide sur une seule ligne, rien d'autre."
)

_PROMPT_HEADER = """Classifie la question en JSON à partir du schéma ci-dessous.

=== SOURCES DISPONIBLES ===
{schema_block}

=== FORMAT DE RÉPONSE (une seule ligne JSON) ===
{{"intent":"list|qa","source":"NOM_TABLE_ou_null","column":"NOM_COLONNE_ou_null","exhaustive":true|false,"filter":{{}} ou null}}

=== RÈGLES ===
1. "intent":"list"  → question qui demande une liste, une énumération, "quels sont", "tous les"
   "intent":"qa"    → question ciblée sur une personne, un nom, une valeur spécifique
2. "source" → NOM EXACT d'une table du schéma, ou null
3. "column" → utilise TOUJOURS la colonne marquée [COLONNE PRINCIPALE] de la table choisie
4. "exhaustive" → true si intent=list, false sinon
5. "filter" → si la question contient un critère précis (un statut, une direction, un département...),
   mets-le avec le NOM DE COLONNE EXACT visible dans le schéma. null si aucun critère.

=== EXEMPLES GÉNÉRÉS DEPUIS LE SCHÉMA ===
{examples_block}

Question: {question}
JSON:"""


class IntentRouter:
    """Classifie chaque question en JSON — 100% dynamique, zéro hardcode."""

    def __init__(self, llm: HFClient, schema: Dict[str, dict], cache_size: int = 256):
        self.llm        = llm
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()

        # Analyser le schéma pour enrichir avec les rôles des colonnes
        analyzer     = SchemaAnalyzer()
        self.schema  = {
            name: (analyzer.analyze(name, info) if not info.get("is_doc") else info)
            for name, info in schema.items()
        }

        # Construire schema_block et examples_block une seule fois
        self._schema_block  = self._build_schema_block()
        self._examples_block = self._build_examples_block()

        logger.info("IntentRouter: %d source(s) analysée(s)", len(self.schema))
        for name, info in self.schema.items():
            if not info.get("is_doc"):
                logger.info("  • %s → entity=%s | filters=%s | name_cols=%s | role=%s",
                            name,
                            info.get("entity_col"),
                            list(info.get("filter_cols", {}).keys()),
                            info.get("name_cols"),
                            info.get("role_col"))

    # ── API publique ──────────────────────────────────────────────────────

    def classify(self, question: str) -> dict:
        key = self._fold(question)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        prompt = _PROMPT_HEADER.format(
            schema_block  = self._schema_block,
            examples_block= self._examples_block,
            question      = question.strip(),
        )
        try:
            raw = self.llm.generate(
                prompt=prompt, system=_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=150,
            )
        except Exception as e:
            logger.warning("IntentRouter: LLM échoué (%s) → fallback", e)
            return self._fallback()

        parsed = self._parse_json(raw)
        result = self._validate(parsed) if parsed else self._fallback()

        logger.info("IntentRouter: intent=%s source=%s col=%s exhaustive=%s filter=%s",
                    result["intent"], result["source"], result["column"],
                    result["exhaustive"], result["filter"])

        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def get_table_info(self, table: str) -> Optional[dict]:
        """Retourne les infos analysées d'une table (entity_col, filter_cols…)."""
        return self.schema.get(table)

    # ── Construction du schema_block ──────────────────────────────────────

    def _build_schema_block(self) -> str:
        """Génère la description du schéma pour le prompt LLM.
        Chaque table est décrite avec :
        - ses colonnes et exemples de valeurs
        - la colonne principale marquée [COLONNE PRINCIPALE]
        - les colonnes de filtre avec leurs valeurs distinctes possibles
        """
        if not self.schema:
            return "(aucune source disponible)"

        lines = []
        for name, info in self.schema.items():
            if info.get("is_doc"):
                lines.append(f"• {name} [DOCUMENT TEXTE] — descriptions, explications")
                continue

            columns    = info.get("columns", [])
            samples    = info.get("samples", {})
            row_count  = info.get("row_count", "?")
            entity_col = info.get("entity_col")
            filter_cols= info.get("filter_cols", {})
            name_cols  = info.get("name_cols", [])
            role_col   = info.get("role_col")

            lines.append(f"• {name} ({row_count} lignes) :")

            for col in columns:
                vals = samples.get(col, [])
                tags = []
                if col == entity_col:
                    tags.append("COLONNE PRINCIPALE")
                if col in filter_cols:
                    # Montrer les valeurs distinctes possibles pour les filtres
                    distinct = filter_cols[col]
                    tags.append(f"FILTRE — valeurs possibles: {', '.join(distinct[:5])}")
                if col in name_cols:
                    tags.append("NOM PERSONNE")
                if col == role_col:
                    tags.append("RÔLE/FONCTION")

                tag_str = f" [{', '.join(tags)}]" if tags else ""
                if vals:
                    ex = ", ".join(f'"{v}"' for v in vals[:3])
                    lines.append(f"    - {col}{tag_str} → ex: {ex}")
                else:
                    lines.append(f"    - {col}{tag_str}")

        return "\n".join(lines)

    # ── Construction des exemples dynamiques ──────────────────────────────

    def _build_examples_block(self) -> str:
        """Génère des exemples JSON réels basés sur le schéma analysé.
        Chaque table produit :
        - 1 exemple de liste simple
        - N exemples de liste avec filtre (un par valeur de filtre catégorielle)
        - 1 exemple de QA personne (si name_cols + role_col existent)
        """
        lines = []
        for name, info in self.schema.items():
            if info.get("is_doc"):
                continue

            entity_col  = info.get("entity_col")
            filter_cols = info.get("filter_cols", {})
            name_cols   = info.get("name_cols", [])
            role_col    = info.get("role_col")
            samples     = info.get("samples", {})

            if not entity_col:
                continue

            table_label = name.replace("_", " ").lower()

            # ── Exemple 1 : liste simple ──────────────────────────────────
            lines.append(f'Q: "Liste des {table_label}"')
            lines.append(json.dumps({
                "intent": "list", "source": name,
                "column": entity_col, "exhaustive": True, "filter": None
            }, ensure_ascii=False))
            lines.append("")

            # ── Exemples filtre : une par valeur distincte ────────────────
            for filter_col, distinct_vals in filter_cols.items():
                # Ne pas générer un exemple pour chaque valeur si trop nombreuses
                for val in distinct_vals[:3]:
                    # Question naturelle : "liste des <table> <val>"
                    lines.append(f'Q: "Liste des {table_label} {val.lower()}"')
                    lines.append(json.dumps({
                        "intent": "list", "source": name,
                        "column": entity_col, "exhaustive": True,
                        "filter": {filter_col: val}
                    }, ensure_ascii=False))
                    lines.append("")

            # ── Exemple QA personne ───────────────────────────────────────
            if name_cols and role_col and filter_cols:
                # Prendre le premier filtre disponible
                first_filter_col = next(iter(filter_cols))
                first_filter_val = filter_cols[first_filter_col][0] \
                    if filter_cols[first_filter_col] else "VALEUR"

                # Question sur le responsable
                lines.append(f'Q: "Qui est le responsable de {first_filter_val} ?"')
                lines.append(json.dumps({
                    "intent": "qa", "source": name,
                    "column": None, "exhaustive": False,
                    "filter": {first_filter_col: first_filter_val}
                }, ensure_ascii=False))
                lines.append("")

                # Question avec entity_col comme filtre (ex: "chef du département X")
                entity_samples = samples.get(entity_col, [])
                if entity_samples:
                    ex_entity = entity_samples[0]
                    lines.append(f'Q: "Qui est le chef de {ex_entity} ?"')
                    lines.append(json.dumps({
                        "intent": "qa", "source": name,
                        "column": None, "exhaustive": False,
                        "filter": {entity_col: ex_entity}
                    }, ensure_ascii=False))
                    lines.append("")

        return "\n".join(lines) if lines else "(aucun exemple généré)"

    # ── Validation et parsing ─────────────────────────────────────────────

    def _validate(self, data: dict) -> dict:
        intent = str(data.get("intent", "qa")).lower()
        if intent not in ("list", "qa", "detail"):
            intent = "qa"

        source = data.get("source")
        if source in ("null", "None", "", None):
            source = None
        if source and source not in self.schema:
            up = str(source).upper()
            source = up if up in self.schema else None

        # Résolution de la colonne
        column = data.get("column")
        if column in ("null", "None", "", None):
            column = None

        if source and not self.schema[source].get("is_doc"):
            info = self.schema[source]
            if column:
                # Vérifier que la colonne existe (insensible casse/accents)
                cols_map = {self._fold(c): c for c in info.get("columns", [])}
                column = cols_map.get(self._fold(column))
            # Si colonne absente ou nulle → utiliser entity_col
            if not column:
                column = info.get("entity_col")

        # Résolution du filtre
        filt = data.get("filter")
        if not isinstance(filt, dict) or not filt:
            filt = None
        elif source and not self.schema[source].get("is_doc"):
            info = self.schema[source]
            cols_map = {self._fold(c): c for c in info.get("columns", [])}
            filt = {
                cols_map.get(self._fold(k), k): v
                for k, v in filt.items()
            }

        exhaustive = (intent == "list") or bool(data.get("exhaustive", False))

        return {
            "intent":     intent,
            "source":     source,
            "column":     column,
            "exhaustive": exhaustive,
            "filter":     filt,
            "confidence": "high",
        }

    def _parse_json(self, raw: str) -> Optional[dict]:
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        candidate = match.group(0) if match else raw.strip()
        for attempt in (candidate, self._repair(candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    @staticmethod
    def _repair(s: str) -> str:
        s = re.sub(r"'", '"', s)
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    @staticmethod
    def _fold(text: str) -> str:
        nfd = unicodedata.normalize("NFD", text.lower().strip())
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    @staticmethod
    def _fallback() -> dict:
        return {
            "intent": "qa", "source": None, "column": None,
            "exhaustive": False, "filter": None, "confidence": "low",
        }
