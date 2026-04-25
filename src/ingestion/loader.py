"""Chargement de documents: PDF, DOCX, TXT, MD, HTML, XLSX, URL."""
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import urllib.request
import urllib.parse


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# XLSX — deux stratégies selon le type de fichier
# ─────────────────────────────────────────────────────────────────────────────

def _is_formations_file(path: str) -> bool:
    """Détecte si le fichier Excel est un plan de formation (vs données RH)."""
    name = Path(path).stem.upper()
    # Correspondance par nom de fichier
    if any(kw in name for kw in ["FORMATION", "GTP"]):
        # Vérifier aussi le contenu de la première cellule
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(max_row=3, values_only=True):
                for cell in row:
                    if cell and "formation" in str(cell).lower():
                        wb.close()
                        return True
            wb.close()
        except Exception:
            pass
    return False


def _load_excel_formations(path: str) -> str:
    """
    Loader spécialisé pour KAM_Formations_GTP.xlsx.

    Produit UNE LIGNE PAR FORMATION avec le préfixe [OBLIGATOIRE] ou [FACULTATIVE].
    Ce format est reconnu directement par le filtre BM25 dans pipeline._try_direct_extract().

    Format de sortie :
        [OBLIGATOIRE] N°01 — Gestion des Déchets et les Aspects Réglementaires | Statut: Obligatoire
        [OBLIGATOIRE] N°02 — Commission Paritaire Hygiène et Sécurité | Statut: Obligatoire
        ...
        [FACULTATIVE] N°01 — Entretien Préventif des Machines d'usinage | Statut: Facultative
        ...

    Pourquoi ce format ?
    - Une ligne = un chunk sémantique complet (le chunker ne coupera pas une formation en deux)
    - Le préfixe [OBLIGATOIRE]/[FACULTATIVE] permet au filtre pipeline de séparer les types
    - Le séparateur "\n\n" entre sections aide le chunker à créer des groupes logiques
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    # Trouver la bonne feuille
    sheet = None
    for name in wb.sheetnames:
        if "formation" in name.lower() or "gtp" in name.lower():
            sheet = wb[name]
            break
    if sheet is None:
        sheet = wb.active

    lines_obligatoire = []
    lines_facultative = []
    header_passed = False
    current_statut = None   # "Obligatoire" | "Facultative"

    for row in sheet.iter_rows(values_only=True):
        cells = [str(c).strip() if c is not None else "" for c in row]

        # Ignorer les lignes vides
        if not any(cells):
            continue

        col0 = cells[0]
        col1 = cells[1] if len(cells) > 1 else ""
        col2 = cells[2] if len(cells) > 2 else ""

        # Détecter les lignes de section (🔴 OBLIGATOIRES / 🟢 FACULTATIVES)
        joined = " ".join(cells).upper()
        if "OBLIGATOIRE" in joined and not col0.isdigit():
            current_statut = "Obligatoire"
            header_passed = False   # réinitialiser pour la prochaine ligne d'en-tête
            continue

        if "FACULTATIV" in joined and not col0.isdigit():
            current_statut = "Facultative"
            header_passed = False
            continue

        # Ignorer les lignes d'en-tête (N° | Intitulé | Statut)
        if col0 in ("N°", "N", "") or "intitul" in col0.lower() or "statut" in col0.lower():
            header_passed = True
            continue

        # Ligne de données : col0 doit être un numéro
        if not col0.isdigit():
            continue

        num = int(col0)
        intitule = col1.strip()
        statut_raw = col2.strip()

        # Déterminer le statut depuis la colonne ou depuis le contexte de section
        if "obligatoire" in statut_raw.lower():
            statut = "Obligatoire"
        elif "facultatif" in statut_raw.lower() or "facultative" in statut_raw.lower():
            statut = "Facultative"
        elif current_statut:
            statut = current_statut
        else:
            statut = "Inconnue"

        # Construire la ligne avec préfixe reconnu par le pipeline
        if statut == "Obligatoire":
            prefix = "[OBLIGATOIRE]"
            line = f"{prefix} N°{num:02d} — {intitule} | Statut: Obligatoire"
            lines_obligatoire.append(line)
        else:
            prefix = "[FACULTATIVE]"
            line = f"{prefix} N°{num:02d} — {intitule} | Statut: Facultative"
            lines_facultative.append(line)

    wb.close()

    # Assembler : section obligatoires + section facultatives séparées par \n\n
    # Le chunker verra des blocs bien distincts
    parts = []
    if lines_obligatoire:
        parts.append(
            "=== FORMATIONS OBLIGATOIRES (" + str(len(lines_obligatoire)) + " formations) ===\n"
            + "\n".join(lines_obligatoire)
        )
    if lines_facultative:
        parts.append(
            "=== FORMATIONS FACULTATIVES (" + str(len(lines_facultative)) + " formations) ===\n"
            + "\n".join(lines_facultative)
        )

    total = len(lines_obligatoire) + len(lines_facultative)
    header = f"PLAN DE FORMATION GTP — {total} FORMATIONS\n\n"

    return header + "\n\n".join(parts)


def _load_excel_generic(path: str) -> str:
    """Loader générique pour les Excel RH (DIRECTION, DEPARTEMENT, SERVICE, POSTE)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"=== Feuille: {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            values = [str(v).strip() for v in row if v is not None]
            if values:
                parts.append(" | ".join(values))
    return "\n".join(parts)


def _load_excel(path: str) -> str:
    """Dispatcher : choisit le bon loader selon le type de fichier Excel."""
    if _is_formations_file(path):
        return _load_excel_formations(path)
    return _load_excel_generic(path)


# ─────────────────────────────────────────────────────────────────────────────
# Autres loaders (inchangés)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


def _load_docx(path: str) -> str:
    from docx import Document as DocxDoc
    doc = DocxDoc(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _load_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


LOADERS = {
    ".pdf":      _load_pdf,
    ".docx":     _load_docx,
    ".doc":      _load_docx,
    ".html":     _load_html,
    ".htm":      _load_html,
    ".txt":      _load_text,
    ".md":       _load_text,
    ".markdown": _load_text,
    ".rst":      _load_text,
    ".csv":      _load_text,
    ".json":     _load_text,
    ".xlsx":     _load_excel,
    ".xls":      _load_excel,
}


def load_document(path: str) -> Document:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Format non supporté: {ext}")

    content = LOADERS[ext](str(p))
    return Document(
        content=content,
        metadata={
            "source":     str(p),
            "filename":   p.name,
            "extension":  ext,
            "size_bytes": p.stat().st_size,
        },
    )


def scrape_url(url: str, timeout: int = 15) -> Document:
    """Scrape une URL HTTP/HTTPS et retourne un Document."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL invalide (http/https requis): {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "RAGBot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_html = resp.read()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
    content = soup.get_text(separator="\n", strip=True)

    safe_name = urllib.parse.quote(url, safe="").replace("%", "_")[:80] + ".url"

    return Document(
        content=content,
        metadata={
            "source":    url,
            "filename":  safe_name,
            "extension": ".url",
            "title":     title,
        },
    )


def load_directory(directory: str) -> List[Document]:
    docs = []
    for file in sorted(Path(directory).rglob("*")):
        if file.is_file() and file.suffix.lower() in LOADERS:
            try:
                doc = load_document(str(file))
                if doc.content.strip():
                    docs.append(doc)
                    print(f"  Chargé: {file.name} ({len(doc.content):,} chars)")
            except Exception as e:
                print(f"  Ignoré {file.name}: {e}")
    return docs
