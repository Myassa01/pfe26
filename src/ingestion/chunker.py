"""Découpage de documents en chunks avec overlap."""
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from .loader import Document


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Détection : est-ce un document formations GTP ?
# ─────────────────────────────────────────────────────────────────────────────

def _is_formations_document(doc: Document) -> bool:
    """
    Retourne True si ce document est le plan de formations GTP.
    Vérifie le nom de fichier ET les marqueurs dans le contenu.
    """
    fname = doc.metadata.get("filename", "").upper()
    # Correspondance par nom
    if any(kw in fname for kw in ["FORMATION", "GTP"]) and fname.endswith((".XLSX", ".XLS")):
        return True
    # Correspondance par contenu (si le loader a bien préfixé)
    content = doc.content
    return "[OBLIGATOIRE]" in content or "[FACULTATIVE]" in content


# ─────────────────────────────────────────────────────────────────────────────
# Chunker spécialisé formations — UN chunk PAR formation
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_formations(doc: Document) -> List[Chunk]:
    """
    Pour le plan de formations, chaque ligne [OBLIGATOIRE]/[FACULTATIVE]
    devient un chunk atomique indépendant.

    Cela garantit que :
    - Le filtre BM25 dans pipeline._try_direct_extract() trouve toujours
      [OBLIGATOIRE] ou [FACULTATIVE] dans le contenu du chunk.
    - Aucune formation n'est coupée à travers deux chunks.
    - Les 66 formations (19 + 47) produisent exactement 66 chunks distincts.

    Un chunk de section header est également créé pour aider la recherche
    sémantique ("formations obligatoires", "formations facultatives").
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", doc.metadata["source"])
    chunks = []
    chunk_index = 0

    current_section = None   # "OBLIGATOIRES" | "FACULTATIVES"
    section_lines: List[str] = []

    def flush_section():
        """Crée un chunk d'en-tête de section (aide le retrieval sémantique)."""
        nonlocal chunk_index
        if not section_lines or not current_section:
            return
        header_content = "\n".join(section_lines)
        chunk_id = f"{safe_name}__section_{chunk_index}"
        chunks.append(Chunk(
            id=chunk_id,
            content=header_content,
            metadata={
                **doc.metadata,
                "chunk_index": chunk_index,
                "chunk_type":  "section_header",
                "section":     current_section,
                "chunk_id":    chunk_id,
            },
        ))
        chunk_index += 1

    for line in doc.content.splitlines():
        line = line.strip()
        if not line:
            continue

        # ── Ligne d'en-tête de section ─────────────────────────────────
        if line.startswith("=== FORMATIONS OBLIGATOIRES"):
            flush_section()
            current_section = "OBLIGATOIRES"
            section_lines = [line]
            continue

        if line.startswith("=== FORMATIONS FACULTATIVES"):
            flush_section()
            current_section = "FACULTATIVES"
            section_lines = [line]
            continue

        # ── Ligne d'en-tête global (PLAN DE FORMATION GTP…) ────────────
        if line.startswith("PLAN DE FORMATION GTP"):
            chunk_id = f"{safe_name}__header_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                content=line,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_index,
                    "chunk_type":  "global_header",
                    "chunk_id":    chunk_id,
                },
            ))
            chunk_index += 1
            continue

        # ── Ligne de formation [OBLIGATOIRE] ou [FACULTATIVE] ──────────
        if line.startswith("[OBLIGATOIRE]") or line.startswith("[FACULTATIVE]"):
            # Chaque formation = 1 chunk atomique
            statut = "Obligatoire" if line.startswith("[OBLIGATOIRE]") else "Facultative"

            # Extraire le numéro et l'intitulé pour les métadonnées
            # Format : [OBLIGATOIRE] N°01 — Titre | Statut: Obligatoire
            num = None
            intitule = ""
            try:
                body = line.split("]", 1)[1].strip()          # "N°01 — Titre | Statut: ..."
                body_no_statut = body.split("|")[0].strip()    # "N°01 — Titre"
                if "—" in body_no_statut:
                    num_part, intitule = body_no_statut.split("—", 1)
                    num_str = re.search(r"\d+", num_part)
                    num = int(num_str.group()) if num_str else None
                    intitule = intitule.strip()
            except Exception:
                pass

            chunk_id = f"{safe_name}__{statut.lower()}_{num or chunk_index:02d}"
            chunks.append(Chunk(
                id=chunk_id,
                content=line,   # contient [OBLIGATOIRE] ou [FACULTATIVE] → filtre BM25 OK
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_index,
                    "chunk_type":  "formation",
                    "statut":      statut,
                    "numero":      num,
                    "intitule":    intitule,
                    "chunk_id":    chunk_id,
                },
            ))
            chunk_index += 1
            continue

        # ── Autres lignes (non reconnues) → ignorer ─────────────────────
        # (ex: lignes de séparation, commentaires)

    # Flush de la dernière section
    flush_section()

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Chunker générique (inchangé)
# ─────────────────────────────────────────────────────────────────────────────

def _split_recursive(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Découpage récursif par séparateurs hiérarchiques."""
    if len(text) <= chunk_size or not separators:
        return [text.strip()] if text.strip() else []

    sep = separators[0]
    rest = separators[1:]

    parts = text.split(sep)
    chunks_list: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks_list.append(current)
            if len(part) > chunk_size:
                sub = _split_recursive(part, chunk_size, rest)
                chunks_list.extend(sub)
                current = ""
            else:
                current = part.strip()

    if current:
        chunks_list.append(current)

    return [c for c in chunks_list if c.strip()]


def _apply_overlap(chunks_list: List[str], overlap: int) -> List[str]:
    """Ajoute un overlap entre chunks consécutifs."""
    if overlap <= 0 or len(chunks_list) <= 1:
        return chunks_list

    result = [chunks_list[0]]
    for i in range(1, len(chunks_list)):
        prev = chunks_list[i - 1]
        prefix = prev[-overlap:] if len(prev) > overlap else prev
        idx = prefix.find(" ")
        if idx > 0:
            prefix = prefix[idx + 1:]
        result.append((prefix + " " + chunks_list[i]).strip() if prefix else chunks_list[i])

    return result


def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 64, **kwargs) -> List[Chunk]:
    """
    Découpe un document en chunks.
    Utilise le chunker formations si le document est un plan de formation,
    sinon le chunker générique récursif.
    """
    # ── Chunker spécialisé formations ──────────────────────────────────
    if _is_formations_document(doc):
        chunks = _chunk_formations(doc)
        print(f"    → {doc.metadata['filename']}: {len(chunks)} chunks formations "
              f"({sum(1 for c in chunks if c.metadata.get('statut') == 'Obligatoire')} oblig. / "
              f"{sum(1 for c in chunks if c.metadata.get('statut') == 'Facultative')} facult.)")
        return chunks

    # ── Chunker générique ───────────────────────────────────────────────
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]
    raw = _split_recursive(doc.content, chunk_size, separators)
    texts = _apply_overlap(raw, overlap)

    chunks = []
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", doc.metadata["source"])
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        chunk_id = f"{safe_name}__{i}"
        chunks.append(Chunk(
            id=chunk_id,
            content=text.strip(),
            metadata={
                "source":       doc.metadata["source"],
                "filename":     doc.metadata["filename"],
                "extension":    doc.metadata["extension"],
                "chunk_index":  i,
                "chunk_total":  len(texts),
                "chunk_id":     chunk_id,
            },
        ))
    return chunks


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    overlap: int = 64,
    **kwargs,                   # absorbe embedding_model et autres params inconnus
) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        if not _is_formations_document(doc):   # formations déjà loggées dans chunk_document
            print(f"  {doc.metadata['filename']}: {len(chunks)} chunks")
    return all_chunks
