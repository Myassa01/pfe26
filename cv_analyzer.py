"""
Module d'analyse de CV via pipeline RAG - Version corrigée et renforcée
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
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
        except ImportError:
            raise RuntimeError("Aucune librairie PDF. Installez : pip install pymupdf")


def extract_text_from_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise RuntimeError("python-docx non installé. Installez : pip install python-docx")


def extract_cv_text(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Format non supporté : {filename}. Formats acceptés : PDF, DOCX, TXT")


# ─────────────────────────────────────────────────────────────
# Validation CV
# ─────────────────────────────────────────────────────────────
MIN_CV_LENGTH = 200

def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    if not cv_text or len(cv_text.strip()) < MIN_CV_LENGTH:
        return {
            "answer": f"⚠️ CV insuffisant ({filename}) — contenu trop limité.",
            "score": 0,
            "poste": "Non analysé",
            "recommended_poste": "Non déterminable — CV incomplet",
            "sources": [],
            "elapsed_seconds": 0.0,
            "years_experience": None,
            "diploma_year": None,
            "filename": filename,
        }
    return None


# ─────────────────────────────────────────────────────────────
# Prompt simplifié et renforcé
# ─────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """
Tu es un ATS strict de Sonatrach (recrutement pétrole & gaz).

POSTE CIBLE :
{poste}

RÉFÉRENTIEL SONATRACH :
{job_context}

CV DU CANDIDAT :
{cv_text}

=== RÈGLES OBLIGATOIRES (respecte-les à la lettre) ===

1. IDENTIFICATION DU DOMAINE
   Identifie clairement le domaine principal du CV :
   - Soudage / Pipeline / Chaudronnerie → "Soudage"
   - Informatique / Développement / Data → "Informatique"
   - Comptabilité / Finance / Gestion → "Comptabilité"
   - Autres domaines : électricité, mécanique, géologie, etc.

2. RÈGLE D'INCOMPATIBILITÉ (LA PLUS IMPORTANTE)
   Si le domaine du CV est différent du domaine du POSTE CIBLE → Score MAXIMUM 2/10 et DOMAINE = Incompatible.
   Exemples :
   - CV Soudeur + Poste Développeur Informatique → Incompatible (0-2/10)
   - CV Comptable + Poste Soudeur → Incompatible (0-2/10)
   - CV Informaticien + Poste Chef d'Équipe Soudeurs → Incompatible (0-2/10)

3. Si les domaines sont compatibles, note normalement (jusqu'à 10/10).

FORMAT DE RÉPONSE OBLIGATOIRE (ne mets rien d'autre) :

**SCORE** : X/10
**DOMAINE** : Compatible / Incompatible
**DÉCISION** : Recommandé / À étudier / Non recommandé
**ATOUTS**
- point court et clair
- point court et clair
**LACUNES**
- point court et clair
**POSTE RECOMMANDÉ** : Titre exact du poste
**ANNÉES_EXPÉRIENCE** : nombre (ex: 12)
**ANNÉE_DIPLOME** : année (ex: 2019) ou 0
"""

def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    poste_label = poste.strip() if poste else "Non précisé"
    return ANALYSIS_PROMPT.format(
        poste=poste_label,
        job_context=job_context or "Aucun référentiel disponible.",
        cv_text=cv_text[:4500],
    )


# ─────────────────────────────────────────────────────────────
# Détection de domaine (anti-hallucination)
# ─────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "soudage": ["soud", "soudeur", "pipeline", "smaw", "tig", "mig", "chaudron", "wps", "asme", "soudage"],
    "informatique": ["développeur", "ingénieur informatic", "python", "java", "sql", "data", "cloud", "devops", "logiciel", "ia", "spark"],
    "comptabilite": ["comptable", "comptabilité", "sap", "sage", "finance", "bilan", "fiscal", "tva"],
    # Ajoute d'autres domaines si nécessaire
}

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    best_domain = "autre"
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def postprocess_score(answer: str, score: Optional[int], cv_text: str, poste: str) -> Optional[int]:
    """Force le score bas en cas d'incompatibilité de domaine."""
    if score is None:
        return score

    cv_domain = detect_domain(cv_text)
    poste_domain = detect_domain(poste) if poste else "autre"

    if (cv_domain != "autre" and poste_domain != "autre" and cv_domain != poste_domain):
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

    search_poste = poste.strip() if poste else ""

    # RAG
    try:
        search_query = f"exigences poste {search_poste} Sonatrach" if search_poste else f"postes Sonatrach {cv_text[:600]}"
        query_embedding = pipeline.embedder.embed_single(search_query)
        dense_results = pipeline.vector_store.search(query_embedding, k=pipeline.config.top_k_dense)
        sparse_results = pipeline.bm25.search(search_query, k=pipeline.config.top_k_sparse) if pipeline.bm25 else []
        
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=pipeline.config.rrf_k)
        top_chunks = fused[:pipeline.config.top_k_after_rerank]

        job_context = "\n\n---\n\n".join(f"[{c['metadata'].get('source', '?')}]\n{c['content']}" for c in top_chunks)
        sources = list({c["metadata"].get("source", "?") for c in top_chunks})
    except Exception as e:
        logger.warning("Erreur RAG : %s", e)
        job_context = ""
        sources = []

    # Prompt + LLM
    prompt = build_analysis_prompt(cv_text, search_poste, job_context)
    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=(
                "Tu es un ATS très strict chez Sonatrach. "
                "Respecte TOUJOURS les règles d'incompatibilité de domaine. "
                "Ne jamais donner plus de 2/10 quand les domaines sont différents."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur LLM : {e}")

    # Post-traitement
    score = _extract_score(answer)
    score = postprocess_score(answer, score, cv_text, search_poste)
    recommended_poste = _extract_recommended_poste(answer)
    years_experience = _extract_years_experience(answer)
    diploma_year = _extract_diploma_year(answer)

    elapsed = round(time.time() - t0, 2)
    displayed_poste = search_poste or recommended_poste or "Non précisé"

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
        r"SCORE\s*:\s*(\d{1,2})\s*/\s*10",
        r"\*\*SCORE\*\*\s*:\s*(\d{1,2})",
        r"(\d{1,2})\s*/\s*10",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = int(m.group(1))
            return val if 0 <= val <= 10 else None
    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
    patterns = [
        r"POSTE\s+RECOMMAND[EÉ]\s*[:\-]\s*([^\n]+)",
        r"\*\*POSTE\s+RECOMMAND[EÉ]\*\*\s*[:\-]\s*([^\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip()
    return None


def _extract_years_experience(text: str) -> Optional[int]:
    m = re.search(r"ANN[EÉ]ES?[_\s]EXP[EÉ]RIENCE[^:]*:\s*(-?\d{1,2})", text, re.I)
    if m:
        val = int(m.group(1))
        return val if -1 <= val <= 50 else None
    return None


def _extract_diploma_year(text: str) -> Optional[int]:
    m = re.search(r"ANN[EÉ]E[_\s]DIPLOM[EÉ][^:]*:\s*((?:19|20)\d{2})", text, re.I)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────
# Tri batch
# ─────────────────────────────────────────────────────────────
def sort_results_with_tiebreaker(results: list) -> list:
    def sort_key(r: dict):
        score = r.get("score") or _extract_score(r.get("answer", "")) or -1
        years = r.get("years_experience") or -1
        diploma = r.get("diploma_year") or 9999
        return (-score, -years, diploma)
    return sorted(results, key=sort_key)
