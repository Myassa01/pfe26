"""
Module d'analyse de CV via pipeline RAG - Version Finale Corrigée
"""
import io
import logging
from typing import Optional
import time
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Extraction texte
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        logger.warning("PyMuPDF absent, fallback pdfplumber")
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()

def extract_text_from_docx(content: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def extract_cv_text(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Format non supporté : {filename}")


# ─────────────────────────────────────────────────────────────
# Validation CV
# ─────────────────────────────────────────────────────────────
MIN_CV_LENGTH = 200

def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    if len(cv_text.strip()) < MIN_CV_LENGTH:
        return {
            "answer": "⚠️ CV insuffisant (contenu trop court)",
            "score": 0,
            "poste": "Non analysé",
            "recommended_poste": "CV incomplet",
            "sources": [],
            "elapsed_seconds": 0.0,
            "years_experience": None,
            "diploma_year": None,
            "filename": filename,
        }
    return None


# ─────────────────────────────────────────────────────────────
# Prompt ultra strict
# ─────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """
Tu es un ATS très strict de Sonatrach.

POSTE CIBLE : {poste}
CV DU CANDIDAT : {cv_text}

RÈGLES ABSOLUES À RESPECTER :
- Identifie d'abord le domaine principal du CV (Soudage, Informatique, Comptabilité...).
- Si le domaine du CV est différent du domaine du POSTE CIBLE → Score maximum 2/10 et DOMAINE = Incompatible.
- Exemples :
  * CV Informatique + Poste Soudeur → Incompatible (0-2/10)
  * CV Soudeur + Poste Développeur → Incompatible (0-2/10)
  * CV Comptable + Poste Soudeur → Incompatible (0-2/10)

Réponds UNIQUEMENT avec ce format exact, rien avant, rien après :

**SCORE** : X/10
**DOMAINE** : Compatible / Incompatible
**DÉCISION** : Recommandé / À étudier / Non recommandé
**ATOUTS**
- point court
- point court
**LACUNES**
- point court
- point court
**POSTE RECOMMANDÉ** : Titre exact
**ANNÉES_EXPÉRIENCE** : nombre
**ANNÉE_DIPLOME** : année ou 0
"""

def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste.strip() if poste else "Non précisé",
        cv_text=cv_text[:4200]
    )


# ─────────────────────────────────────────────────────────────
# Détection domaine + post-processing
# ─────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "soudage": ["soud", "soudeur", "pipeline", "smaw", "tig", "mig", "chaudron", "soudage"],
    "informatique": ["développeur", "ingénieur informatic", "python", "sql", "data", "cloud", "devops", "ia", "spark"],
    "comptabilite": ["comptable", "comptabilité", "sap", "sage", "finance", "bilan"],
}

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    best, best_score = "autre", 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best = domain
    return best

def postprocess_score(answer: str, score: Optional[int], cv_text: str, poste: str) -> Optional[int]:
    if score is None:
        return None
    if poste:
        cv_domain = detect_domain(cv_text)
        poste_domain = detect_domain(poste)
        if cv_domain != "autre" and poste_domain != "autre" and cv_domain != poste_domain:
            return min(score, 2)
    return score


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────
def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str, filename: str = "CV") -> dict:
    t0 = time.time()

    validation_error = validate_cv_text(cv_text, filename)
    if validation_error:
        return validation_error

    # RAG simplifié
    try:
        search_query = f"exigences poste {poste} Sonatrach" if poste else cv_text[:600]
        query_embedding = pipeline.embedder.embed_single(search_query)
        dense_results = pipeline.vector_store.search(query_embedding, k=8)
        job_context = "\n\n---\n\n".join(c['content'] for c in dense_results[:5])
        sources = list({c["metadata"].get("source", "?") for c in dense_results})
    except Exception as e:
        logger.warning("Erreur RAG: %s", e)
        job_context = ""
        sources = []

    prompt = build_analysis_prompt(cv_text, poste, job_context)

    answer = pipeline.llm.generate(
        prompt=prompt,
        system="Tu es un ATS strict. Respecte EXACTEMENT le format demandé. Applique la règle d'incompatibilité de domaine sans exception. Pas de texte supplémentaire.",
        temperature=0.0,
        max_tokens=1000,
    )

    # Extraction + correction
    score = _extract_score(answer)
    score = postprocess_score(answer, score, cv_text, poste)
    recommended_poste = _extract_recommended_poste(answer)
    years_experience = _extract_years_experience(answer)
    diploma_year = _extract_diploma_year(answer)

    elapsed = round(time.time() - t0, 2)
    displayed_poste = poste or recommended_poste or "Non précisé"

    return {
        "answer": answer,
        "score": score,
        "poste": displayed_poste,
        "recommended_poste": recommended_poste or "Non précisé",
        "sources": sources,
        "elapsed_seconds": elapsed,
        "years_experience": years_experience,
        "diploma_year": diploma_year,
        "filename": filename,
    }


# ─────────────────────────────────────────────────────────────
# Extractors
# ─────────────────────────────────────────────────────────────
def _extract_score(text: str) -> Optional[int]:
    patterns = [
        r"SCORE\s*[:\-]?\s*(\d{1,2})\s*/\s*10",
        r"(\d{1,2})\s*/\s*10",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 10:
                return val
    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
    m = re.search(r"POSTE\s+RECOMMAND[EÉ]\s*[:\-]?\s*([^\n]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_years_experience(text: str) -> Optional[int]:
    m = re.search(r"ANN[EÉ]ES?[_ ]?EXP[EÉ]RIENCE[^:]*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_diploma_year(text: str) -> Optional[int]:
    m = re.search(r"ANN[EÉ]E[_ ]?DIPLOM[EÉ][^:]*[:\-]?\s*(\d{4})", text, re.IGNORECASE)
    return int(m.group(1)) if m else None
