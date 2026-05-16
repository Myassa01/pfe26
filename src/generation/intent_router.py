"""Routage intelligent des requêtes : découverte de schéma + classification LLM.

Remplace les listes de mots-clés hard-codés du pipeline par :
1. SchemaDiscovery : scan automatique des fichiers du dossier docs (colonnes, échantillons)
2. IntentRouter   : appel LLM léger qui retourne {intent, source, column, exhaustive, filter}

Changements vs version précédente
──────────────────────────────────
- Le bloc d'exemples du prompt est **généré dynamiquement** à partir du schéma
  réel (noms de tables, noms de colonnes, valeurs d'échantillon réelles).
  → Plus aucun nom hard-codé (SERVICE, DIRECTION, SHORT_LIBELLE_DIRECTION…).
- Le nombre d'échantillons par colonne passe à 5 (au lieu de 3) pour aider le
  LLM à comprendre les colonnes techniques (ex: CHANTIER = nom de service).
- `_build_schema_block` inclut le row_count réel (fourni par StructuredQueryEngine).
- `_build_dynamic_examples` génère 4 types d'exemples pour chaque table connue :
    • liste exhaustive de la table
    • liste filtrée (si une colonne de filtre plausible existe)
    • question ciblée (qa) avec filtre
    • comptage
- `_validate` est renforcé : fallback fuzzy sur les noms de colonnes du filtre,
  et avertissement explicite si source inconnue (au lieu d'un silent None).
- Cache LRU conservé (OrderedDict, taille configurable).
"""

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm import HFClient
from ..ingestion.loader import _detect_header_row

logger = logging.getLogger(__name__)


# ── SCHEMA DISCOVERY ──────────────────────────────────────────────────────────

class SchemaDiscovery:
    """Scanne le dossier de documents et extrait, pour chaque fichier :
       - les colonnes (Excel) ou un flag is_doc=True (Word/PDF/etc.)
       - quelques échantillons de valeurs par colonne (max `max_samples`, courts)
    Le résultat sert à injecter le schéma dans le prompt du classifieur."""

    EXCEL_EXTS = {".xlsx", ".xls"}
    DOC_EXTS   = {".docx", ".doc", ".pdf", ".txt", ".md", ".html", ".htm"}

    def __init__(self, docs_dir: str, max_samples: int = 5, max_sample_len: int = 40):
        self.docs_dir      = docs_dir
        self.max_samples   = max_samples      # augmenté : 5 au lieu de 3
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

        logger.info(
            "SchemaDiscovery: %d sources détectées (%s)",
            len(schema), ", ".join(schema.keys()),
        )
        return schema

    @staticmethod
    def _normalize_stem(fname: str) -> str:
        """SERVICE (1).xlsx → SERVICE  ;  KAM_Formations.xlsx → KAM_FORMATIONS"""
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
            header_row = _detect_header_row(ws)
            headers: List[str] = [
                str(cell.value).strip() if cell.value else ""
                for cell in ws[header_row]
            ]
            headers = [h for h in headers if h]

            samples: Dict[str, List[str]] = {h: [] for h in headers}
            for i, row in enumerate(ws.iter_rows(min_row=header_row + 1, values_only=True)):
                if i >= 200:
                    break
                for h, v in zip(headers, row):
                    if v is None or len(samples[h]) >= self.max_samples:
                        continue
                    s = str(v).strip()[: self.max_sample_len]
                    if s and s not in samples[h]:
                        samples[h].append(s)
            wb.close()
            return {
                "columns": headers,
                "samples": {h: samples[h] for h in headers if samples[h]},
                "is_doc":  False,
                "filename": path.name,
            }
        except Exception as e:
            logger.warning("SchemaDiscovery: échec scan %s: %s", path.name, e)
            return None


# ── PROMPTS ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Tu es un classifieur de requêtes en français. "
    "Tu retournes UNIQUEMENT un objet JSON valide, rien d'autre."
)

# Le bloc {examples_block} est généré dynamiquement depuis le schéma réel.
_PROMPT_TEMPLATE = """\
Classifie la question utilisateur en JSON selon le schéma ci-dessous.

Sources disponibles (chaque ligne = 1 table, avec ses colonnes et des exemples de valeurs) :
{schema_block}

Format JSON attendu (strict, une seule ligne) :
{{"intent":"list|detail|qa","source":"<NOM_SOURCE_OU_null>","column":"<NOM_COLONNE_OU_null>","exhaustive":true|false,"filter":null}}

Règles :
- "intent":"list"   → la question demande une énumération ("liste", "tous", "donne-moi les", "quels sont", "combien").
- "intent":"detail" → la question demande une explication ("explique", "détails", "décris").
- "intent":"qa"     → toute question ciblée ("qui est X", "quel est le X de Y").
- "source"  : EXACTEMENT un nom de table parmi la liste ci-dessus, ou null.
- "column"  : EXACTEMENT un nom de colonne de la table choisie, ou null si la question vise toute la ligne.
- "exhaustive" : true si intent="list", false sinon.
- "filter"  : dict {{"NOM_COLONNE_REEL":"valeur"}} pour les contraintes, ou null.
  → Utilise UNIQUEMENT des noms de colonnes présents dans le schéma ci-dessus.

Exemples basés sur le schéma ci-dessus :
{examples_block}

Question : {question}
JSON :"""


# ── INTENT ROUTER ──────────────────────────────────────────────────────────────

class IntentRouter:
    """Classifie une question utilisateur via le LLM en s'appuyant sur le schéma découvert.

    Toutes les références aux tables et colonnes dans le prompt sont construites
    dynamiquement depuis `schema` — aucun nom hard-codé dans le code source.
    """

    def __init__(self, llm: HFClient, schema: Dict[str, dict], cache_size: int = 256):
        self.llm        = llm
        self.schema     = schema
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, dict]" = OrderedDict()

        # Construits une seule fois, réutilisés à chaque classify()
        self._schema_block   = self._build_schema_block(schema)
        self._examples_block = self._build_dynamic_examples(schema)

        logger.info(
            "IntentRouter: %d source(s) dans le schéma (%s)",
            len(schema), ", ".join(schema.keys()),
        )

    # ── API publique ───────────────────────────────────────────────────────────

    def classify(self, question: str) -> dict:
        key = self._normalize_question(question)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        prompt = _PROMPT_TEMPLATE.format(
            schema_block=self._schema_block,
            examples_block=self._examples_block,
            question=question.strip(),
        )
        try:
            raw = self.llm.generate(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=150,
            )
        except Exception as e:
            logger.warning("IntentRouter: appel LLM échoué (%s) — fallback qa", e)
            return self._fallback()

        parsed = self._parse_json(raw)
        result = self._validate(parsed) if parsed else self._fallback()
        logger.info(
            "IntentRouter: intent=%s source=%s column=%s exhaustive=%s filter=%s conf=%s",
            result["intent"], result["source"], result["column"],
            result["exhaustive"], result["filter"], result["confidence"],
        )

        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    # ── Construction du bloc schéma ────────────────────────────────────────────

    def _build_schema_block(self, schema: Dict[str, dict]) -> str:
        """Expose chaque source avec colonnes + échantillons de valeurs.

        Ex pour une table EMPLOYES :
            * EMPLOYES (120 lignes) :
                - NOM → ex: "Dupont", "Martin", "Leroy"
                - PRENOM → ex: "Jean", "Marie"
                - POSTE → ex: "Ingénieur", "Technicien"
        """
        if not schema:
            return "(aucune source disponible)"

        lines = []
        for name, info in schema.items():
            if info.get("is_doc"):
                lines.append(f"* {name} (document texte) — descriptions, explications")
                continue

            row_count = info.get("row_count", "?")
            cols      = info.get("columns", [])
            samples   = info.get("samples",  {})

            lines.append(f"* {name} ({row_count} lignes) :")
            for col in cols:
                vals = samples.get(col, [])
                if vals:
                    sample_str = ", ".join(f'"{v}"' for v in vals[:5])
                    lines.append(f"    - {col} → ex: {sample_str}")
                else:
                    lines.append(f"    - {col}")

        return "\n".join(lines)

    # ── Génération dynamique des exemples ──────────────────────────────────────

    def _build_dynamic_examples(self, schema: Dict[str, dict]) -> str:
        """Génère des exemples de classification *à partir du schéma réel*.

        Pour chaque table tabulaire on produit 4 exemples :
          1. Liste exhaustive
          2. Liste filtrée (si une colonne de filtre plausible est trouvée)
          3. Question ciblée (qa) avec filtre
          4. Comptage

        Cela garantit que le LLM voit uniquement des noms de tables et de
        colonnes qui existent réellement dans les fichiers chargés.
        """
        if not schema:
            return "(aucun exemple disponible — schéma vide)"

        lines: List[str] = []

        for table_name, info in schema.items():
            if info.get("is_doc"):
                # Exemple générique pour les documents
                lines.append(
                    f'Q: "Explique {table_name.lower()}"\n'
                    f'JSON: {{"intent":"detail","source":"{table_name}",'
                    f'"column":null,"exhaustive":false,"filter":null}}'
                )
                continue

            cols    = info.get("columns", [])
            samples = info.get("samples", {})
            if not cols:
                continue

            # ── Exemple 1 : liste exhaustive ──────────────────────────────
            lines.append(
                f'Q: "Liste des {table_name.lower()}"\n'
                f'JSON: {{"intent":"list","source":"{table_name}",'
                f'"column":null,"exhaustive":true,"filter":null}}'
            )

            # ── Choisir une colonne de filtre + une valeur d'exemple ──────
            filter_col, filter_val = self._pick_filter_column(cols, samples, table_name)

            if filter_col and filter_val:
                # ── Exemple 2 : liste filtrée ─────────────────────────────
                lines.append(
                    f'Q: "{table_name.lower()} de {filter_val}"\n'
                    f'JSON: {{"intent":"list","source":"{table_name}",'
                    f'"column":null,"exhaustive":true,'
                    f'"filter":{{"{filter_col}":"{filter_val}"}}}}'
                )

                # ── Exemple 3 : question ciblée (qa) ──────────────────────
                # On choisit une colonne cible différente de la colonne filtre
                target_col = self._pick_target_column(cols, samples, exclude=filter_col)
                if target_col:
                    lines.append(
                        f'Q: "Quel est le {target_col.lower()} pour {filter_val} ?"\n'
                        f'JSON: {{"intent":"qa","source":"{table_name}",'
                        f'"column":"{target_col}","exhaustive":false,'
                        f'"filter":{{"{filter_col}":"{filter_val}"}}}}'
                    )

            # ── Exemple 4 : comptage ──────────────────────────────────────
            lines.append(
                f'Q: "Combien de {table_name.lower()} ?"\n'
                f'JSON: {{"intent":"list","source":"{table_name}",'
                f'"column":null,"exhaustive":true,"filter":null}}'
            )

            lines.append("")  # ligne vide entre tables

        return "\n".join(lines).strip()

    def _pick_filter_column(
        self,
        cols: List[str],
        samples: Dict[str, List[str]],
        table_name: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Choisit la colonne la plus appropriée pour servir de filtre dans les
        exemples, en se basant sur les valeurs réelles.

        Préfère les colonnes dont les valeurs sont :
          - courtes (codes, noms courts) → bon filtre
          - peu nombreuses relativement au total → catégorielle
        Exclut les colonnes numériques pures (IDs, matricules).
        """
        best_col: Optional[str] = None
        best_val: Optional[str] = None
        best_score: float = -1.0

        for col in cols:
            vals = [v for v in samples.get(col, []) if v and len(v) >= 2]
            if not vals:
                continue

            # Ignorer les colonnes purement numériques
            if all(v.replace(".", "").replace(",", "").replace("-", "").isdigit()
                   for v in vals):
                continue

            avg_len = sum(len(v) for v in vals) / len(vals)

            # Pénaliser les valeurs trop longues (phrases) ou trop courtes (1 char)
            if avg_len > 40 or avg_len < 2:
                continue

            # Bonus si le nom de la colonne suggère une FK / catégorie
            col_up = col.upper()
            bonus = 1.0
            for kw in ("DIRECTION", "DEPARTEMENT", "SERVICE", "TYPE", "CATEGORIE",
                       "STATUT", "STATUS", "ACTIVITE", "ZONE", "REGION", "SITE",
                       "WILAYA", "UNITE", "GROUPE"):
                if kw in col_up:
                    bonus = 2.0
                    break

            # Pénaliser les colonnes qui semblent être des identifiants primaires
            # de la table elle-même (ex: table DIRECTION → colonne DIRECTION)
            table_stem = table_name.upper().replace("_", "")
            col_stem   = col_up.replace("_", "")
            if table_stem in col_stem or col_stem in table_stem:
                bonus *= 0.5

            score = bonus / avg_len  # préfère les valeurs courtes
            if score > best_score:
                best_score = score
                best_col   = col
                # Prend la première valeur non vide comme exemple
                best_val   = vals[0]

        return best_col, best_val

    @staticmethod
    def _pick_target_column(
        cols: List[str],
        samples: Dict[str, List[str]],
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        """Choisit une colonne cible pour les exemples de type 'qa'.

        Préfère les colonnes avec des valeurs lisibles (nom, libellé, prénom…).
        """
        NAME_KWS = ("NOM", "LIBELLE", "PRENOM", "INTITULE", "DESIGNATION",
                    "TITRE", "LABEL", "CHANTIER")

        for col in cols:
            if col == exclude:
                continue
            col_up = col.upper()
            for kw in NAME_KWS:
                if kw in col_up:
                    return col

        # Fallback : première colonne avec des valeurs texte non-numériques
        for col in cols:
            if col == exclude:
                continue
            vals = samples.get(col, [])
            if vals and not all(
                v.replace(".", "").replace(",", "").replace("-", "").isdigit()
                for v in vals
            ):
                return col

        return None

    # ── Parsing / validation ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_question(q: str) -> str:
        q = q.lower().strip()
        for src, dst in [
            ("é","e"),("è","e"),("ê","e"),("ë","e"),
            ("à","a"),("â","a"),("ä","a"),
            ("î","i"),("ï","i"),("ô","o"),("ö","o"),
            ("ù","u"),("û","u"),("ü","u"),("ç","c"),
        ]:
            q = q.replace(src, dst)
        return re.sub(r"\s+", " ", q)

    def _parse_json(self, raw: str) -> Optional[dict]:
        if not raw:
            return None
        match     = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        candidate = match.group(0) if match else raw.strip()
        for attempt in (candidate, self._repair_json(candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    @staticmethod
    def _repair_json(s: str) -> str:
        s = re.sub(r"'", '"', s)
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    def _validate(self, data: dict) -> dict:
        # ── intent ────────────────────────────────────────────────────────
        intent = str(data.get("intent", "qa")).lower()
        if intent not in ("list", "detail", "qa"):
            intent = "qa"

        # ── source ────────────────────────────────────────────────────────
        source = data.get("source")
        if source in (None, "null", "None", ""):
            source = None
        else:
            source = self._resolve_source(str(source))

        # ── column ────────────────────────────────────────────────────────
        column = data.get("column")
        if column in (None, "null", "None", ""):
            column = None
        elif source and not self.schema[source].get("is_doc"):
            column = self._resolve_column(source, str(column))

        # ── exhaustive ────────────────────────────────────────────────────
        # exhaustive = True  →  la question demande une liste complète
        # On distingue ensuite deux sous-cas dans le pipeline :
        #   • source tabulaire connue  → bypass DuckDB (0 chunk RAG)
        #   • source documentaire/null → RAG normal, top_k modéré (pas 200)
        exhaustive = bool(data.get("exhaustive", False)) or intent == "list"

        # exhaustive_structured : True seulement si la source est une table SQL
        # connue et NON documentaire → le pipeline peut faire le bypass DuckDB.
        # Si la source est un doc texte ou null, on garde exhaustive=True mais
        # exhaustive_structured=False pour que le pipeline utilise top_k normal.
        is_structured_source = (
            source is not None
            and not self.schema.get(source, {}).get("is_doc", True)
        )
        exhaustive_structured = exhaustive and is_structured_source

        # ── filter ────────────────────────────────────────────────────────
        filt = data.get("filter")
        if not isinstance(filt, dict) or not filt:
            filt = None
        elif source and not self.schema[source].get("is_doc"):
            cols_up = {
                c.upper(): c
                for c in self.schema[source].get("columns", [])
            }
            resolved_filt: Dict[str, str] = {}
            for k, v in filt.items():
                real_col = self._fuzzy_resolve_column(str(k), cols_up)
                if real_col:
                    resolved_filt[real_col] = str(v)
                else:
                    logger.warning(
                        "IntentRouter._validate: colonne de filtre inconnue '%s' "
                        "ignorée pour la table %s (colonnes: %s)",
                        k, source, ", ".join(cols_up.keys()),
                    )
            filt = resolved_filt or None

        return {
            "intent":                intent,
            "source":                source,
            "column":                column,
            "exhaustive":            exhaustive,
            "exhaustive_structured": exhaustive_structured,
            "filter":                filt,
            "confidence":            "high",
        }

    # ── Helpers de résolution ──────────────────────────────────────────────────

    def _resolve_source(self, name: str) -> Optional[str]:
        """Résout un nom de source : exact → upper → fuzzy partiel."""
        if name in self.schema:
            return name
        up = name.upper().strip()
        if up in self.schema:
            return up
        # Match partiel : le LLM peut avoir tronqué ou rajouté un suffixe
        for key in self.schema:
            if key in up or up in key:
                logger.debug("IntentRouter: source '%s' résolue en '%s' (fuzzy)", name, key)
                return key
        logger.warning(
            "IntentRouter._validate: source '%s' inconnue (schéma: %s)",
            name, ", ".join(self.schema.keys()),
        )
        return None

    def _resolve_column(self, source: str, col_name: str) -> Optional[str]:
        """Résout un nom de colonne pour une source tabulaire."""
        cols_up = {
            c.upper(): c
            for c in self.schema[source].get("columns", [])
        }
        return self._fuzzy_resolve_column(col_name, cols_up)

    @staticmethod
    def _fuzzy_resolve_column(
        col_name: str,
        cols_up: Dict[str, str],   # {UPPER: original}
    ) -> Optional[str]:
        """
        Résolution en 3 passes :
          1. Exact (après normalisation upper)
          2. Containment : la clé contient col ou col contient la clé
          3. Préfixe commun (≥ 4 chars)
        """
        target = col_name.upper().strip()

        # Passe 1 — exact
        if target in cols_up:
            return cols_up[target]

        # Passe 2 — containment
        for key, original in cols_up.items():
            if target in key or key in target:
                return original

        # Passe 3 — préfixe commun ≥ 4 chars
        for key, original in cols_up.items():
            min_len = min(len(target), len(key))
            if min_len >= 4 and target[:min_len] == key[:min_len]:
                return original

        return None

    @staticmethod
    def _fallback() -> dict:
        return {
            "intent":                "qa",
            "source":                None,
            "column":                None,
            "exhaustive":            False,
            "exhaustive_structured": False,
            "filter":                None,
            "confidence":            "low",
        }
