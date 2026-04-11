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


def _split_recursive(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Découpage récursif par séparateurs hiérarchiques."""
    if len(text) <= chunk_size or not separators:
        return [text.strip()] if text.strip() else []

    sep = separators[0]
    rest = separators[1:]

    parts = text.split(sep)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                sub = _split_recursive(part, chunk_size, rest)
                chunks.extend(sub)
                current = ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Ajoute un overlap entre chunks consécutifs."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Prend les derniers `overlap` caractères du chunk précédent
        prefix = prev[-overlap:] if len(prev) > overlap else prev
        # Coupe au premier espace pour éviter les coupures en milieu de mot
        idx = prefix.find(" ")
        if idx > 0:
            prefix = prefix[idx + 1:]
        result.append((prefix + " " + chunks[i]).strip() if prefix else chunks[i])

    return result


def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]
    raw = _split_recursive(doc.content, chunk_size, separators)
    texts = _apply_overlap(raw, overlap)

    chunks = []
    # Crée un ID safe pour ChromaDB (pas de caractères spéciaux)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", doc.metadata["source"])
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        chunk_id = f"{safe_name}__{i}"
        chunks.append(Chunk(
            id=chunk_id,
            content=text.strip(),
            metadata={
                "source": doc.metadata["source"],
                "filename": doc.metadata["filename"],
                "extension": doc.metadata["extension"],
                "chunk_index": i,
                "chunk_total": len(texts),
                "chunk_id": chunk_id,
            },
        ))
    return chunks


def chunk_documents(docs: List[Document], chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        print(f"  {doc.metadata['filename']}: {len(chunks)} chunks")
    return all_chunks
