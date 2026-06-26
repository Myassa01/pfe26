"""
Module d'analyse de CV - Version Finale Renforcée
"""
import io
import logging
from typing import Optional
import time
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Extraction texte (inchangée)
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
# Validation
# ─────────────────────────────────────────────────────────────
MIN_CV_LENGTH = 200

def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    if len(cv_text.strip()) < MIN_CV_LENGTH:
        return {"answer": "⚠️ CV trop court", "score": 0, "poste": "Non analysé", "recommended_poste": "CV incomplet", "sources": [], "elapsed_seconds": 0.0, "years_experience": None, "diploma_year": None, "filename": filename}
    return None

# ─────────────────────────────────────────────────────────────
# Prompt ULTRA RENFORCÉ
# ─────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """
Tu es un ATS très strict de Sonatrach.

POSTE CIBLE: {poste}
RÉFÉRENTIEL: {job_context}
CV: {cv_text}

RÈGLES STRICTES :
- Identifie d'abord le domaine du CV (Soudage / Informatique / Comptabilité...).
- Si domaine différent du poste cible → SCORE MAX 2/10 et DOMAINE = Incompatible.
- Respecte EXACTEMENT ce format. Rien d'autre.

**FORMAT OBLIGATOIRE (copie-colle exactement) :**

**SCORE** : X/10
**DOMAINE** : Compatible / Incompatible
**DÉCISION** : Recommandé / À étudier / Non recommandé
**ATOUTS**
- point très court
- point très court
**LACUNES**
- point très court
- point très court
**POSTE RECOMMANDÉ** : Titre exact
**ANNÉES_EXPÉRIENCE** : nombre
**ANNÉE_DIPLOME** : année ou 0

Exemple correct pour un soudeur sur poste soudeur :
**SCORE** : 9/10
**DOMAINE** : Compatible
**DÉCISION** : Recommandé
**ATOUTS**
- 14 ans en soudage pipelines
- Maîtrise SMAW, TIG, MIG
**LACUNES**
- Pas d'expérience en diamètre >36"
**POSTE RECOMMANDÉ** : Chef D'Equipe Soudeurs
**ANNÉES_EXPÉRIENCE** : 14
**ANNÉE_DIPLOME** : 2005

Ne mets aucun texte avant **SCORE**, aucun *LACUNES*, aucun commentaire supplémentaire.
"""

def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste.strip() if poste else "Non précisé",
        job_context=job_context or "Aucun",
        cv_text=cv_text[:4200]
    )

# ─────────────────────────────────────────────────────────────
# Détection domaine
# ─────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "soudage": ["soud", "soudeur", "pipeline", "smaw", "tig", "mig", "chaudron"],
    "informatique": ["développeur", "ingénieur informatic", "python", "sql", "data", "cloud", "devops"],
    "comptabilite": ["comptable", "comptabilité", "sap", "sage", "finance"],
}

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    best, best_score = "autre", 0
    for domain, kws in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > best_score:
            best_score, best = score, domain
    return best

def postprocess_score(answer: str, score: Optional[int], cv_text: str, poste: str) -> Optional[int]:
    if score is None:
        return None
    if poste:
        cv_d = detect_domain(cv_text)
        poste_d = detect_domain(poste)
        if cv_d != "autre" and poste_d != "autre" and cv_d != poste_d:
            return min(score, 2)
    return score

# ─────────────────────────────────────────────────────────────
# Analyse
# ─────────────────────────────────────────────────────────────
def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str, filename: str = "CV") -> dict:
    t0 = time.time()
    if err := validate_cv_text(cv_text, filename):
        return err

    # RAG simplifié
    try:
        search_query = f"exigences {poste} Sonatrach" if poste else cv_text[:500]
        query_embedding = pipeline.embedder.embed_single(search_query)
        dense = pipeline.vector_store.search(query_embedding, k=8)
        job_context = "\n\n---\n\n".join(f"{c['content']}" for c in dense[:5])
        sources = list({c["metadata"].get("source", "?") for c in dense})
    except Exception:
        job_context = sources = []

    prompt = build_analysis_prompt(cv_text, poste, job_context)

    answer = pipeline.llm.generate(
        prompt=prompt,
        system="Réponds UNIQUEMENT avec le format exact demandé. Pas de texte supplémentaire, pas d'astérisques hors format, pas de LACUNAGES.",
        temperature=0.0,
        max_tokens=1200,
    )

    score = _extract_score(answer)
    score = postprocess_score(answer, score, cv_text, poste)
    recommended = _extract_recommended_poste(answer)
    years = _extract_years_experience(answer)
    diploma = _extract_diploma_year(answer)

    return {
        "answer": answer,
        "score": score,
        "poste": poste or recommended or "Non précisé",
        "recommended_poste": recommended or "Non précisé",
        "sources": sources,
        "elapsed_seconds": round(time.time() - t0, 2),
        "years_experience": years,
        "diploma_year": diploma,
        "filename": filename,
    }

# ─────────────────────────────────────────────────────────────
# Extractors améliorés
# ─────────────────────────────────────────────────────────────
def _extract_score(text: str) -> Optional[int]:
    for pat in [
        r"SCORE\s*:\s*(\d{1,2})\s*/\s*10",
        r"(\d{1,2})\s*/\s*10",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            val = int(m.group(1))
            return val if 0 <= val <= 10 else None
    return None

def _extract_recommended_poste(text: str) -> Optional[str]:
    m = re.search(r"POSTE\s+RECOMMAND[EÉ]\s*[:\-]\s*([^\n]+)", text, re.I)
    return m.group(1).strip() if m else None

def _extract_years_experience(text: str) -> Optional[int]:
    m = re.search(r"ANN[EÉ]ES?[_ ]?EXP[EÉ]RIENCE[^:]*:\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else None

def _extract_diploma_year(text: str) -> Optional[int]:
    m = re.search(r"ANN[EÉ]E[_ ]?DIPLOM[EÉ][^:]*:\s*(\d{4})", text, re.I)
    return int(m.group(1)) if m else None
