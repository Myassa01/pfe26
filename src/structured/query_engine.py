"""StructuredQueryEngine — interroge les fichiers Excel via DuckDB en mémoire.

Architecture :
    1. Au démarrage, scanne docs_dir et charge chaque .xlsx dans une table DuckDB.
    2. Auto-détection de la ligne d'en-tête (réutilise loader._detect_header_row)
       pour gérer les Excel avec titre/métadonnées avant le tableau.
    3. Expose des méthodes haut niveau (list_values, filter_rows, count_rows)
       qui construisent du SQL paramétré — JAMAIS de concaténation directe de
       valeurs utilisateur dans les requêtes (protection injection SQL).

Pourquoi DuckDB plutôt que pandas / SQLite :
    - Lecture directe des .xlsx via l'extension `excel` (pas de conversion).
    - SQL standard, performances excellentes sur dizaines de milliers de lignes.
    - In-memory par défaut, zéro fichier à gérer.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StructuredQueryEngine:
    EXCEL_EXTS = {".xlsx", ".xls"}

    def __init__(self, docs_dir: str):
        import duckdb
        self._duckdb = duckdb
        self.conn = duckdb.connect(":memory:")
        # Mapping nom_normalisé (UPPERCASE, sans extension) → métadonnées de la table
        # {table_name: {"filename": str, "columns": [str], "row_count": int}}
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.docs_dir = docs_dir
        self._load_all(docs_dir)

    # ── Chargement ────────────────────────────────────────────────────────

    def _load_all(self, docs_dir: str) -> None:
        if not Path(docs_dir).exists():
            logger.warning("StructuredQueryEngine: docs_dir introuvable: %s", docs_dir)
            return
        loaded = 0
        for file in sorted(Path(docs_dir).rglob("*")):
            if not file.is_file() or file.suffix.lower() not in self.EXCEL_EXTS:
                continue
            try:
                self._load_excel(file)
                loaded += 1
            except Exception as e:
                logger.warning("  ⚠ Skip %s: %s", file.name, e)
        logger.info("StructuredQueryEngine: %d table(s) chargée(s) en DuckDB (%s)",
                    loaded, ", ".join(self.tables.keys()) or "—")

    def _load_excel(self, path: Path) -> None:
        """Charge un Excel comme table DuckDB. Utilise openpyxl pour respecter
        l'auto-détection du header (cas KAM_Formations_GTP) puis insère via
        DuckDB."""
        from ..ingestion.loader import _detect_header_row
        import openpyxl

        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
        try:
            ws = wb.active
            header_row = _detect_header_row(ws)
            headers: List[str] = []
            for cell in ws[header_row]:
                headers.append(str(cell.value).strip() if cell.value else "")

            # Colonnes : on garde celles non vides, on les rend SQL-safe
            keep_idx = [i for i, h in enumerate(headers) if h]
            sql_headers = [self._sql_ident(headers[i]) for i in keep_idx]
            if not sql_headers:
                logger.warning("  ⚠ %s: aucune colonne valide", path.name)
                return

            # Désambiguïser les doublons (DuckDB refuse les colonnes en double)
            seen: Dict[str, int] = {}
            unique_headers: List[str] = []
            for h in sql_headers:
                if h in seen:
                    seen[h] += 1
                    unique_headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    unique_headers.append(h)

            rows: List[List[Optional[str]]] = []
            for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
                values = [row[i] if i < len(row) else None for i in keep_idx]
                # Toutes les valeurs en str pour cohérence (les types Excel sont incertains)
                values = [None if v is None else str(v).strip() for v in values]
                if any(v for v in values):  # ignorer lignes complètement vides
                    rows.append(values)
        finally:
            wb.close()

        if not rows:
            logger.warning("  ⚠ %s: aucune donnée", path.name)
            return

        table_name = self._normalize_stem(path.name)
        sql_table = self._sql_ident(table_name)

        # CREATE TABLE avec colonnes VARCHAR
        cols_def = ", ".join(f'"{c}" VARCHAR' for c in unique_headers)
        self.conn.execute(f'CREATE OR REPLACE TABLE "{sql_table}" ({cols_def})')

        # INSERT en bulk — placeholders paramétrés (pas de concaténation user input)
        placeholders = ", ".join(["?"] * len(unique_headers))
        self.conn.executemany(
            f'INSERT INTO "{sql_table}" VALUES ({placeholders})',
            rows,
        )

        self.tables[table_name] = {
            "filename": path.name,
            "columns": unique_headers,
            "row_count": len(rows),
            "sql_table": sql_table,
        }
        logger.info("  ✓ %s → table %s (%d lignes, %d colonnes)",
                    path.name, sql_table, len(rows), len(unique_headers))

    # ── Helpers normalisation ─────────────────────────────────────────────

    @staticmethod
    def _normalize_stem(fname: str) -> str:
        """SERVICE (1).xlsx → SERVICE  ;  KAM_Formations_GTP.xlsx → KAM_FORMATIONS_GTP"""
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        stem = stem.upper().strip()
        stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
        stem = re.sub(r"\s*_\d+\s*$", "", stem)
        return stem.strip()

    @staticmethod
    def _sql_ident(name: str) -> str:
        """Rend un nom de colonne/table sûr pour SQL (alphanumérique + _)."""
        # Retire accents
        nfd = unicodedata.normalize("NFD", name)
        ascii_name = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        # Garde alphanum et _, remplace le reste par _
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", ascii_name).strip("_").upper()
        if not cleaned:
            cleaned = "COL"
        if cleaned[0].isdigit():
            cleaned = "C_" + cleaned
        return cleaned

    @staticmethod
    def _fold(text: str) -> str:
        """Normalise pour comparaison floue (minuscules + sans accents)."""
        nfd = unicodedata.normalize("NFD", str(text))
        ascii_text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return ascii_text.lower().strip()

    def _resolve_column(self, table: str, column: Optional[str]) -> Optional[str]:
        """Résout un nom de colonne (LLM) vers le nom SQL réel (insensible casse/accents)."""
        if not column:
            return None
        cols = self.tables[table]["columns"]
        target = self._fold(column)
        target_sql = self._sql_ident(column)
        for c in cols:
            if self._fold(c) == target or c == target_sql:
                return c
        return None

    # ── API publique ──────────────────────────────────────────────────────

    def has_table(self, name: str) -> bool:
        return self._normalize_stem(name) in self.tables \
            or name in self.tables

    def schema(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le schéma sans la connexion DuckDB (pour SchemaDiscovery)."""
        return {
            name: {
                "columns": info["columns"],
                "filename": info["filename"],
                "row_count": info["row_count"],
                "is_doc": False,
            }
            for name, info in self.tables.items()
        }

    def samples(self, table: str, max_per_col: int = 3, max_len: int = 40) -> Dict[str, List[str]]:
        """Échantillons de valeurs distinctes par colonne (pour le prompt LLM)."""
        table = self._normalize_stem(table) if table not in self.tables else table
        if table not in self.tables:
            return {}
        sql_table = self.tables[table]["sql_table"]
        out: Dict[str, List[str]] = {}
        for col in self.tables[table]["columns"]:
            try:
                rows = self.conn.execute(
                    f'SELECT DISTINCT "{col}" FROM "{sql_table}" '
                    f'WHERE "{col}" IS NOT NULL LIMIT {max_per_col}'
                ).fetchall()
                vals = [str(r[0])[:max_len] for r in rows if r[0]]
                if vals:
                    out[col] = vals
            except Exception:
                continue
        return out

    def list_values(
        self,
        table: str,
        column: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
        distinct: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retourne les valeurs d'une colonne (ou les lignes complètes si column=None).

        column=None → retourne toutes les colonnes (List[Dict[col, value]])
        column="X"  → retourne List[Dict["content": value, "metadata": {...}]]
                       pour cohérence avec le format chunk attendu par pipeline.

        filters : appliqués en LOWER(col) LIKE LOWER(?), insensible accents/casse.
        """
        table = self._normalize_stem(table) if table not in self.tables else table
        if table not in self.tables:
            return []
        sql_table = self.tables[table]["sql_table"]
        meta_filename = self.tables[table]["filename"]

        # Résoudre les noms de colonnes (filter + select) vs schéma réel
        resolved_filters: List[tuple] = []  # [(sql_col, value), ...]
        for fkey, fval in (filters or {}).items():
            sql_col = self._resolve_column(table, fkey)
            if sql_col and fval:
                resolved_filters.append((sql_col, str(fval)))

        sql_col = self._resolve_column(table, column) if column else None

        # SELECT clause
        if sql_col:
            select_clause = f'"{sql_col}"'
        else:
            cols = self.tables[table]["columns"]
            select_clause = ", ".join(f'"{c}"' for c in cols)

        distinct_kw = "DISTINCT " if distinct else ""

        # WHERE clause — toujours paramétrée
        where_parts = []
        params: List[str] = []
        if sql_col:
            where_parts.append(f'"{sql_col}" IS NOT NULL')
            where_parts.append(f'TRIM("{sql_col}") <> \'\'')
        for col, val in resolved_filters:
            where_parts.append(
                f'strip_accents(LOWER(CAST("{col}" AS VARCHAR))) '
                f'LIKE strip_accents(LOWER(?))'
            )
            params.append(f"%{val}%")

        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql = f'SELECT {distinct_kw}{select_clause} FROM "{sql_table}"{where_clause}'

        try:
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()
            col_names = [d[0] for d in cursor.description]
        except Exception as e:
            # Fallback si strip_accents indispo (DuckDB ancien) → comparaison lower simple
            logger.warning("StructuredQueryEngine: requête SQL échouée (%s) — fallback lower", e)
            return self._list_values_fallback(table, column, filters, distinct)

        results: List[Dict[str, Any]] = []
        for row in rows:
            if sql_col:
                val = row[0]
                if val is None or not str(val).strip():
                    continue
                results.append({
                    "content": str(val).strip(),
                    "metadata": {"filename": meta_filename, "table": table, "column": sql_col},
                })
            else:
                # Reconstruire un texte type chunk "[SOURCE] K1: V1 | K2: V2"
                pairs = [f"{c}: {v}" for c, v in zip(col_names, row) if v is not None and str(v).strip()]
                if not pairs:
                    continue
                content = f"[{table}] " + " | ".join(pairs)
                results.append({
                    "content": content,
                    "metadata": {"filename": meta_filename, "table": table},
                })

        logger.info("  ⟹ DuckDB: %d résultat(s) [table=%s, col=%s, filters=%s]",
                    len(results), table, sql_col or "ALL", filters or "-")
        return results

    def _list_values_fallback(self, table, column, filters, distinct):
        """Si strip_accents indispo, on rappatrie tout et on filtre en Python."""
        sql_table = self.tables[table]["sql_table"]
        meta_filename = self.tables[table]["filename"]
        cols = self.tables[table]["columns"]
        rows = self.conn.execute(f'SELECT * FROM "{sql_table}"').fetchall()
        sql_col = self._resolve_column(table, column) if column else None

        resolved_filters = []
        for fkey, fval in (filters or {}).items():
            rc = self._resolve_column(table, fkey)
            if rc and fval:
                resolved_filters.append((rc, self._fold(fval)))

        results = []
        seen_distinct: set = set()
        for row in rows:
            kv = dict(zip(cols, row))
            ok = True
            for fc, fv in resolved_filters:
                actual = kv.get(fc)
                if actual is None or fv not in self._fold(str(actual)):
                    ok = False
                    break
            if not ok:
                continue
            if sql_col:
                val = kv.get(sql_col)
                if val is None or not str(val).strip():
                    continue
                key = self._fold(val)
                if distinct and key in seen_distinct:
                    continue
                seen_distinct.add(key)
                results.append({"content": str(val).strip(),
                                "metadata": {"filename": meta_filename, "table": table, "column": sql_col}})
            else:
                pairs = [f"{c}: {v}" for c, v in kv.items() if v is not None and str(v).strip()]
                if pairs:
                    results.append({"content": f"[{table}] " + " | ".join(pairs),
                                    "metadata": {"filename": meta_filename, "table": table}})
        return results

    def count_rows(self, table: str, filters: Optional[Dict[str, str]] = None) -> int:
        """Compte les lignes (avec filtres optionnels)."""
        rows = self.list_values(table, column=None, filters=filters, distinct=False)
        return len(rows)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
