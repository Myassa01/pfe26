"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
Version améliorée :
- Mode analyse ciblée (poste fourni)
- Mode auto-détection (poste vide)
- Proposition de plusieurs postes recommandés
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Extraction texte
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    """Extraction texte PDF."""
    try:
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        return text.strip()

    except ImportError:
        logger.warning("PyMuPDF absent, fallback pdfplumber")

        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                ).strip()

        except ImportError:
            raise RuntimeError(
                "Aucune librairie PDF disponible. "
                "Installez : pip install pymupdf"
            )


def extract_text_from_docx(content: bytes) -> str:
    """Extraction DOCX."""
    try:
        from docx import Document

        doc = Document(io.BytesIO(content))

        return "\n".join(
            p.text for p in doc.paragraphs
        ).strip()

    except ImportError:
        raise RuntimeError(
            "python-docx non installé. "
            "Installez : pip install python-docx"
        )


def extract_cv_text(content: bytes, filename: str) -> str:
    """Dispatcher extraction."""
    name = filename.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)

    elif name.endswith(".docx"):
        return extract_text_from_docx(content)

    elif name.endswith(".txt"):
        return content.decode(
            "utf-8",
            errors="replace"
        ).strip()

    raise ValueError(
        f"Format non supporté : {filename}. "
        "Formats acceptés : PDF, DOCX, TXT"
    )


# ─────────────────────────────────────────────────────────────
# Validation CV
# ─────────────────────────────────────────────────────────────

MIN_CV_LENGTH = 200


def validate_cv_text(
    cv_text: str,
    filename: str
) -> Optional[dict]:
    """
    Vérifie que le CV contient suffisamment d'informations.
    """

    if not cv_text or len(cv_text.strip()) < MIN_CV_LENGTH:

        return {
            "answer": (
                f"⚠️ CV insuffisant ({filename}) — "
                f"contenu trop limité.\n\n"
                f"Le fichier contient seulement "
                f"{len(cv_text.strip())} caractères "
                f"(minimum requis : {MIN_CV_LENGTH})."
            ),
            "score": 0,
            "poste": "Non analysé",
            "recommended_postes": [],
            "sources": [],
            "elapsed_seconds": 0.0,
            "years_experience": None,
            "diploma_year": None,
            "filename": filename,
        }

    return None


# ─────────────────────────────────────────────────────────────
# Prompt analyse ciblée
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH senior chez Sonatrach.

Mission :
Évaluer OBJECTIVEMENT le candidat pour le poste demandé.

════════════════════════════════════════
POSTE CIBLE
════════════════════════════════════════

{poste}

════════════════════════════════════════
EXIGENCES DU POSTE
════════════════════════════════════════

{job_context}

════════════════════════════════════════
CV DU CANDIDAT
════════════════════════════════════════

{cv_text}

════════════════════════════════════════
RÈGLES
════════════════════════════════════════

- Être strict
- Ne jamais inventer
- Évaluer uniquement les éléments présents dans le CV
- Si domaine incompatible → score max = 2/10

════════════════════════════════════════
FORMAT OBLIGATOIRE
════════════════════════════════════════

**SCORE** : X/10
**DOMAINE** : Compatible / Partiellement compatible / Incompatible
**DÉCISION** : Recommandé / À étudier / Non recommandé

**ATOUTS**
- xxx
- xxx

**LACUNES**
- xxx
- xxx

**POSTE RECOMMANDÉ**
- xxx

**ANNÉES_EXPÉRIENCE** : entier ou -1
**ANNÉE_DIPLOME** : année ou 0
"""


# ─────────────────────────────────────────────────────────────
# Prompt auto-détection
# ─────────────────────────────────────────────────────────────

AUTO_DETECTION_PROMPT = """
Tu es un expert RH senior chez Sonatrach.

Le candidat n’a PAS précisé de poste cible.

Ta mission :
1. Identifier le domaine réel du CV
2. Déterminer les métiers compatibles
3. Proposer 2 à 3 postes adaptés
4. Donner une évaluation réaliste

IMPORTANT :
- Ne jamais inventer
- Être strict
- Utiliser uniquement le CV

════════════════════════════
CV DU CANDIDAT
════════════════════════════

{cv_text}

════════════════════════════
FORMAT OBLIGATOIRE
════════════════════════════

**DOMAINE PRINCIPAL** : xxx

**NIVEAU GLOBAL**
Débutant / Junior / Confirmé

**POSTES RECOMMANDÉS**
1. xxx
2. xxx
3. xxx

**COMPÉTENCES CLÉS**
- xxx
- xxx
- xxx

**POINTS FORTS**
- xxx
- xxx

**LACUNES**
- xxx
- xxx

**ANNÉES_EXPÉRIENCE** : entier ou -1
**ANNÉE_DIPLOME** : année ou 0
"""


# ─────────────────────────────────────────────────────────────
# Construction prompt
# ─────────────────────────────────────────────────────────────

def build_analysis_prompt(
    cv_text: str,
    poste: str,
    job_context: str
) -> str:

    # ─────────────────────────────────────
    # MODE AUTO-DETECTION
    # ─────────────────────────────────────
    if not poste or not poste.strip():

        return AUTO_DETECTION_PROMPT.format(
            cv_text=cv_text[:4000]
        )

    # ─────────────────────────────────────
    # MODE ANALYSE CIBLÉE
    # ─────────────────────────────────────
    return ANALYSIS_PROMPT.format(
        poste=poste,
        job_context=job_context or (
            "Aucune exigence spécifique trouvée."
        ),
        cv_text=cv_text[:4000],
    )


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(
    pipeline,
    cv_text: str,
    poste: str,
    filename: str = "CV"
) -> dict:

    import time

    t0 = time.time()

    # ─────────────────────────────────────
    # Validation
    # ─────────────────────────────────────

    validation_error = validate_cv_text(
        cv_text,
        filename
    )

    if validation_error:
        return validation_error

    search_poste = poste.strip() if poste else ""

    # ─────────────────────────────────────
    # MODE AVEC POSTE
    # ─────────────────────────────────────

    search_query = None

    if search_poste:

        search_query = (
            f"exigences compétences diplômes "
            f"requis poste {search_poste}"
        )

    # ─────────────────────────────────────
    # RAG
    # ─────────────────────────────────────

    job_context = ""
    sources = []

    if search_query:

        try:
            query_embedding = (
                pipeline.embedder.embed_single(
                    search_query
                )
            )

            dense_results = (
                pipeline.vector_store.search(
                    query_embedding,
                    k=pipeline.config.top_k_dense,
                )
            )

            sparse_results = []

            if pipeline.bm25:
                sparse_results = (
                    pipeline.bm25.search(
                        search_query,
                        k=pipeline.config.top_k_sparse,
                    )
                )

            from src.retrieval.hybrid_search import (
                reciprocal_rank_fusion
            )

            fused = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=pipeline.config.rrf_k,
            )

            top_chunks = fused[
                :pipeline.config.top_k_after_rerank
            ]

            # ─────────────────────────────
            # Reranker
            # ─────────────────────────────

            if pipeline.reranker and top_chunks:

                pairs = [
                    (search_query, c["content"])
                    for c in top_chunks
                ]

                scores = (
                    pipeline.reranker.model.predict(
                        pairs
                    )
                )

                for c, s in zip(top_chunks, scores):
                    c["rerank_score"] = float(s)

                top_chunks = sorted(
                    top_chunks,
                    key=lambda x: x.get(
                        "rerank_score",
                        0
                    ),
                    reverse=True,
                )

            # ─────────────────────────────
            # Context
            # ─────────────────────────────

            job_context = "\n\n---\n\n".join(
                f"[{c['metadata'].get('source', '?')}]\n"
                f"{c['content']}"
                for c in top_chunks
            )

            sources = list({
                c["metadata"].get(
                    "source",
                    "?"
                )
                for c in top_chunks
            })

        except Exception as e:

            logger.warning(
                "Erreur recherche RAG : %s",
                e
            )

            job_context = ""
            sources = []

    # ─────────────────────────────────────
    # Prompt
    # ─────────────────────────────────────

    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
    )

    # ─────────────────────────────────────
    # Appel LLM
    # ─────────────────────────────────────

    try:

        answer = pipeline.llm.generate(
            prompt=prompt,
            system=(
                "Tu es un expert RH Sonatrach. "
                "Réponds uniquement en français. "
                "Sois direct, strict et concis."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )

    except Exception as e:

        raise RuntimeError(
            f"Erreur analyse LLM : {e}"
        )

    # ─────────────────────────────────────
    # Extraction infos
    # ─────────────────────────────────────

    score = _extract_score(answer)

    recommended_postes = (
        _extract_recommended_postes(answer)
    )

    years_experience = (
        _extract_years_experience(answer)
    )

    diploma_year = (
        _extract_diploma_year(answer)
    )

    elapsed = round(
        time.time() - t0,
        2
    )

    # ─────────────────────────────────────
    # Retour
    # ─────────────────────────────────────

    return {
        "answer": answer,
        "score": score,
        "poste": (
            search_poste
            or "Auto-détection"
        ),
        "recommended_postes": (
            recommended_postes
        ),
        "sources": sources,
        "elapsed_seconds": elapsed,
        "years_experience": (
            years_experience
        ),
        "diploma_year": diploma_year,
        "filename": filename,
    }


# ─────────────────────────────────────────────────────────────
# Tri batch
# ─────────────────────────────────────────────────────────────

def sort_results_with_tiebreaker(
    results: list
) -> list:

    def sort_key(r: dict):

        score = r.get("score")

        if score is None:
            score = _extract_score(
                r.get("answer", "")
            )

        score = (
            score
            if score is not None
            else -1
        )

        years_exp = r.get(
            "years_experience"
        )

        years_exp = (
            years_exp
            if (
                years_exp is not None
                and years_exp >= 0
            )
            else -1
        )

        diploma_year = r.get(
            "diploma_year"
        )

        diploma_year = (
            diploma_year
            if (
                diploma_year
                and diploma_year > 0
            )
            else 9999
        )

        return (
            -score,
            -years_exp,
            diploma_year,
        )

    return sorted(
        results,
        key=sort_key
    )


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _extract_score(
    text: str
) -> Optional[int]:

    import re

    patterns = [
        r"SCORE[^:\n]*:\s*(\d{1,2})\s*/\s*10",
        r"(\d{1,2})\s*/\s*10",
    ]

    for pat in patterns:

        m = re.search(
            pat,
            text,
            re.IGNORECASE
        )

        if m:

            val = int(m.group(1))

            if 0 <= val <= 10:
                return val

    return None


def _extract_recommended_postes(
    text: str
) -> list:

    import re

    postes = []

    patterns = [
        r"POSTES?\s+RECOMMAND[ÉE]S?(.*?)(?:COMP[ÉE]TENCES|POINTS FORTS|LACUNES)"
    ]

    for pat in patterns:

        m = re.search(
            pat,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not m:
            continue

        block = m.group(1)

        found = re.findall(
            r"(?:\d+[\).\-\s]|[-•])\s*(.+)",
            block
        )

        for p in found:

            p = p.strip()

            if len(p) > 3:
                postes.append(p)

    # fallback analyse ciblée
    if not postes:

        m = re.search(
            r"POSTE\s+RECOMMAND[ÉE]\s*[:\-]?\s*(.+)",
            text,
            re.IGNORECASE
        )

        if m:

            p = m.group(1).strip()

            if len(p) > 3:
                postes.append(p)

    return postes[:3]


def _extract_years_experience(
    text: str
) -> Optional[int]:

    import re

    patterns = [
        r"ANN[EÉ]ES?[_\s]EXP[EÉ]RIENCE[^:\n]*:\s*(-?\d{1,2})",
    ]

    for pat in patterns:

        m = re.search(
            pat,
            text,
            re.IGNORECASE
        )

        if m:

            val = int(m.group(1))

            if -1 <= val <= 50:
                return val

    return None


def _extract_diploma_year(
    text: str
) -> Optional[int]:

    import re

    patterns = [
        r"ANN[EÉ]E[_\s]DIPLOM[EÉ][^:\n]*:\s*((?:19|20)\d{2})",
    ]

    for pat in patterns:

        m = re.search(
            pat,
            text,
            re.IGNORECASE
        )

        if m:
            return int(m.group(1))

    return None
