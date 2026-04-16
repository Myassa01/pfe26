"""Découpage de documents en chunks avec overlap — taille mesurée en tokens."""
import logging
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from .loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]


def _make_token_counter(model_name: Optional[str] = None) -> Callable[[str], int]:
    """Retourne une fonction qui compte les tokens d'un texte.
    Utilise le tokenizer du modèle d'embedding si disponible, sinon fallback sur len()."""
    if model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                f"sentence-transformers/{model_name}" if "/" not in model_name else model_name
            )
            return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback : estimation ~1 token ≈ 4 caractères (approximation courante)
    return lambda text: len(text) // 4


def _split_recursive(
    text: str, chunk_size: int, separators: List[str], count_fn: Callable[[str], int]
) -> List[str]:
    """Découpage récursif par séparateurs hiérarchiques (taille en tokens)."""
    if count_fn(text) <= chunk_size or not separators:
        return [text.strip()] if text.strip() else []

    sep = separators[0]
    rest = separators[1:]

    parts = text.split(sep)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if count_fn(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if count_fn(part) > chunk_size:
                sub = _split_recursive(part, chunk_size, rest, count_fn)
                chunks.extend(sub)
                current = ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def _apply_overlap(chunks: List[str], overlap_tokens: int, count_fn: Callable[[str], int]) -> List[str]:
    """Ajoute un overlap entre chunks consécutifs (mesuré en tokens)."""
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Extrait les derniers mots du chunk précédent jusqu'à atteindre overlap_tokens
        words = prev.split()
        prefix_words = []
        for w in reversed(words):
            prefix_words.insert(0, w)
            if count_fn(" ".join(prefix_words)) >= overlap_tokens:
                break
        prefix = " ".join(prefix_words)
        result.append((prefix + " " + chunks[i]).strip() if prefix else chunks[i])

    return result


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64,
    embedding_model: Optional[str] = None,
) -> List[Chunk]:
    count_fn = _make_token_counter(embedding_model)
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]
    raw = _split_recursive(doc.content, chunk_size, separators, count_fn)
    texts = _apply_overlap(raw, overlap, count_fn)

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
                "source": doc.metadata["source"],
                "filename": doc.metadata["filename"],
                "extension": doc.metadata["extension"],
                "chunk_index": i,
                "chunk_total": len(texts),
                "chunk_id": chunk_id,
            },
        ))
    return chunks


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    overlap: int = 64,
    embedding_model: Optional[str] = None,
) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, overlap, embedding_model)
        all_chunks.extend(chunks)
        logger.info("  %s: %d chunks", doc.metadata['filename'], len(chunks))
    return all_chunks
