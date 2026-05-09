"""Routage intelligent des requêtes : découverte de schéma + classification LLM."""
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from .llm import HFClient
from ..ingestion.loader import _detect_header_row

logger = logging.getLogger(__name__)


# ── SCHEMA DISCOVERY ─────────────────────────────────────────────────────────

class SchemaDiscovery:
    EXCEL_EXTS = {".xlsx", ".xls"}
    DOC_EXTS   = {".docx", ".doc", ".pdf", ".txt", ".md", ".html", ".htm"}

    def __init__(self, docs_dir: str, max_samples: int = 3, max_sample_len: int = 40):
        self.docs_dir       = docs_dir
        self.max_samples    = max_samples
        self.max_sample_len = max_sample_len

    def scan(self) -> Dict[str, dict]:
        schema: Dict[str, dict] = {}
        if not Path(self.docs_dir).exists():
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
                schema[stem] = {"columns": [], "samples": {}, "is_doc": True,
                                "filename": file.name}
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
            header_row = _detect_header_row(ws)
            headers: List[str] = []
            for cell in ws[header_row]:
                headers.append(str(cell.value).strip() if cell.value else "")
            headers = [h for h in headers if h]
            samples: Dict[str, List[str]] = {h: [] for h in headers}
            scanned = 0
            for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
                scanned += 1
                if scanned > 200:
                    break
                for h, v in zip(headers, row):
                    if v is None or len(samples[h]) >= self.max_samples:
                        continue
                    s = str(v).strip()[:self.max_sample_len]
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


# ── INTENT ROUTER ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Tu es un classifieur de requêtes en français. "
    "Tu retournes UNIQUEMENT un objet JSON valide, rien d'autre."
)

_PROMPT_TEMPLATE = """Classifie la question utilisateur en JSON selon le schéma ci-dessous.

Sources disponibles :
{schema_block}

Format JSON (une seule ligne) :
{{"intent":"list|detail|qa","source":"<NOM_SOURCE_OU_null>","column":"<NOM_COLONNE_OU_null>","exhaustive":true|false,"filter":null}}

Règles :
- intent=list si la question demande une liste → exhaustive=true
- intent=qa pour les questions ciblées sur une personne ou un élément précis
- source: EXACTEMENT un nom de table ou null
- column: colonne marquée ⭐ ou colonne avec valeurs lisibles. Jamais SHORT_*/ID/MATRICULE.
- filter: {{"COLONNE":"valeur"}} si contrainte explicite dans la question, sinon null

Exemples :
{dynamic_examples}

Question: {question}
JSON:"""


class IntentRouter:
    def __init__(
        self,
        llm: HFClient,
        schema: Dict[str, dict],
        cache_size: int = 256,
        cache_path: str = "./data/intent_cache.json",  # ← persistant sur disque
    ):
        self.llm        = llm
        self.schema     = schema
        self.cache_size = cache_size
        self.cache_path = Path(cache_path)

        # Charge le cache depuis le disque (survit aux redémarrages)
        self._cache: OrderedDict = self._load_cache()

        self._schema_block     = self._build_schema_block(schema)
        self._dynamic_examples = self._build_dynamic_examples(schema)

    # ── Cache persistant ─────────────────────────────────────────────────────

    def _load_cache(self) -> OrderedDict:
        """Charge le cache depuis le disque au démarrage."""
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                logger.info("IntentRouter: cache chargé depuis disque (%d entrées)", len(data))
                return OrderedDict(data)
            except Exception as e:
                logger.warning("IntentRouter: cache illisible (%s) → démarrage à vide", e)
        return OrderedDict()

    def _save_cache(self) -> None:
        """Sauvegarde le cache sur disque après chaque nouvelle classification."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(dict(self._cache), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("IntentRouter: sauvegarde cache échouée (%s)", e)

    def invalidate_cache(self) -> None:
        """Vide le cache (à appeler après un changement de schéma / nouvel Excel)."""
        self._cache.clear()
        if self.cache_path.exists():
            self.cache_path.unlink(missing_ok=True)
        logger.info("IntentRouter: cache invalidé.")

    # ── Public ───────────────────────────────────────────────────────────────

    def classify(self, question: str) -> dict:
        key = self._normalize_question(question)
        if key in self._cache:
            self._cache.move_to_end(key)
            logger.debug("IntentRouter: cache hit pour %r", question[:60])
            return self._cache[key]

        prompt = _PROMPT_TEMPLATE.format(
            schema_block=self._schema_block,
            dynamic_examples=self._dynamic_examples,
            question=question.strip(),
        )
        try:
            raw = self.llm.generate(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=120,
            )
        except Exception as e:
            logger.warning("IntentRouter: LLM échoué (%s) → fallback", e)
            return self._fallback()

        parsed = self._parse_json(raw)
        result = self._validate(parsed) if parsed else self._fallback()
        logger.info("IntentRouter: intent=%s source=%s column=%s exhaustive=%s conf=%s",
                    result["intent"], result["source"], result["column"],
                    result["exhaustive"], result["confidence"])

        # Sauvegarde dans le cache + persist sur disque
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        self._save_cache()

        return result

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_question(q: str) -> str:
        q = q.lower().strip()
        for src, dst in [("é","e"),("è","e"),("ê","e"),("ë","e"),
                         ("à","a"),("â","a"),("ä","a"),("î","i"),("ï","i"),
                         ("ô","o"),("ö","o"),("ù","u"),("û","u"),("ü","u"),("ç","c")]:
            q = q.replace(src, dst)
        return re.sub(r"\s+", " ", q)

    def _build_schema_block(self, schema: Dict[str, dict]) -> str:
        if not schema:
            return "(aucune source disponible)"
        lines = []
        for name, info in schema.items():
            if info.get("is_doc"):
                lines.append(f"* {name} (document texte)")
                continue
            cols      = info.get("columns", [])
            samples   = info.get("samples", {})
            row_count = info.get("row_count", "?")
            label_col = info.get("label_column")
            lines.append(f"* {name} ({row_count} lignes) :")
            for col in cols:
                star = " ⭐ COLONNE RECOMMANDÉE" if col == label_col else ""
                vals = samples.get(col, [])
                if vals:
                    sample_str = ", ".join(f'"{v}"' for v in vals[:3])
                    lines.append(f"    - {col}{star} → ex: {sample_str}")
                else:
                    lines.append(f"    - {col}{star}")
        return "\n".join(lines)

    def _build_dynamic_examples(self, schema: Dict[str, dict]) -> str:
        if not schema:
            return "(aucun exemple)"
        lines = []
        for table_name, info in schema.items():
            if info.get("is_doc"):
                continue
            label_col   = info.get("label_column")
            columns     = info.get("columns", [])
            samples     = info.get("samples", {})
            display_col = label_col or self._pick_display_col(columns, samples)
            if not display_col:
                continue
            table_label = table_name.replace("_", " ").lower()

            # Exemple liste
            lines.append(f'Q: "Liste des {table_label}"')
            lines.append(
                f'{{"intent":"list","source":"{table_name}",'
                f'"column":"{display_col}","exhaustive":true,"filter":null}}'
            )
            lines.append("")

            # Exemple avec filtre
            filter_col = self._pick_filter_col(columns, samples, exclude=display_col)
            if filter_col:
                sample_vals = samples.get(filter_col, [])
                filter_val  = sample_vals[0] if sample_vals else "VALEUR"
                lines.append(f'Q: "{table_label} filtrés par {filter_col}={filter_val}"')
                lines.append(
                    f'{{"intent":"list","source":"{table_name}",'
                    f'"column":"{display_col}","exhaustive":true,'
                    f'"filter":{{"{filter_col}":"{filter_val}"}}}}'
                )
                lines.append("")

                # Exemple QA personne
                lines.append(f'Q: "Qui est responsable de {filter_val} ?"')
                lines.append(
                    f'{{"intent":"qa","source":"{table_name}",'
                    f'"column":null,"exhaustive":false,'
                    f'"filter":{{"{filter_col}":"{filter_val}"}}}}'
                )
                lines.append("")

        return "\n".join(lines) if lines else "(aucun exemple généré)"

    def _parse_json(self, raw: str) -> Optional[dict]:
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
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
        intent = str(data.get("intent", "qa")).lower()
        if intent not in ("list", "detail", "qa"):
            intent = "qa"

        source = data.get("source")
        if source in ("null", "None", "", None):
            source = None
        if source and source not in self.schema:
            up = str(source).upper()
            source = up if up in self.schema else None

        column = data.get("column")
        if column in ("null", "None", "", None):
            column = None
        if column and source and not self.schema[source].get("is_doc"):
            cols_upper = {c.upper(): c for c in self.schema[source].get("columns", [])}
            column = cols_upper.get(str(column).upper())

        exhaustive = bool(data.get("exhaustive", False)) or intent == "list"

        filt = data.get("filter")
        if not isinstance(filt, dict) or not filt:
            filt = None
        elif source:
            cols_upper = {c.upper(): c for c in self.schema[source].get("columns", [])}
            filt = {cols_upper.get(str(k).upper(), k): v for k, v in filt.items()}

        return {
            "intent":     intent,
            "source":     source,
            "column":     column,
            "exhaustive": exhaustive,
            "filter":     filt,
            "confidence": "high",
        }

    @staticmethod
    def _fallback() -> dict:
        return {
            "intent": "qa", "source": None, "column": None,
            "exhaustive": False, "filter": None, "confidence": "low",
        }

    @staticmethod
    def _pick_display_col(columns: List[str], samples: Dict[str, List[str]]) -> Optional[str]:
        for prefix in ["LIBELLE_", "INTITULE_", "DESIGNATION_", "NOM_", "TITRE_"]:
            for col in columns:
                if col.upper().startswith(prefix) and not col.upper().startswith("SHORT_"):
                    return col
        best, best_len = None, 0
        skip = {"ID", "AFFECTATION", "MATRICULE", "OBSERVATION", "AFFECT_PAR"}
        for col in columns:
            if col.upper() in skip or col.upper().startswith(("SHORT_", "ID_", "NUM_")):
                continue
            vals = samples.get(col, [])
            avg  = sum(len(v) for v in vals) / len(vals) if vals else 0
            if avg > best_len:
                best, best_len = col, avg
        return best

    @staticmethod
    def _pick_filter_col(columns: List[str], samples: Dict[str, List[str]],
                         exclude: Optional[str]) -> Optional[str]:
        candidates = []
        for col in columns:
            if col == exclude:
                continue
            cup = col.upper()
            if any(cup.startswith(p) for p in ("SHORT_", "STATUT", "TYPE_", "CATEGORIE")):
                candidates.insert(0, col)
                continue
            vals = samples.get(col, [])
            if not vals:
                continue
            avg_len    = sum(len(v) for v in vals) / len(vals)
            is_numeric = all(v.replace(".", "").replace(",", "").isdigit() for v in vals)
            if 2 < avg_len < 20 and not is_numeric:
                candidates.append(col)
        return candidates[0] if candidates else None
