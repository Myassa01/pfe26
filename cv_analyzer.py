"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
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
        return "\n".join(p.text for p in doc.paragraphs).strip()
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
        return content.decode("utf-8", errors="replace").strip()

    raise ValueError(
        f"Format non supporté : {filename}. "
        "Formats acceptés : PDF, DOCX, TXT"
    )


# ─────────────────────────────────────────────────────────────
# Validation CV
# ─────────────────────────────────────────────────────────────

MIN_CV_LENGTH = 200


def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    """
    Vérifie que le CV contient suffisamment d'informations.
    Retourne un dict d'erreur si invalide, None sinon.
    """
    if not cv_text or len(cv_text.strip()) < MIN_CV_LENGTH:
        return {
            "answer": (
                f"⚠️ CV insuffisant ({filename}) — le contenu est trop limité "
                f"pour permettre une analyse fiable.\n\n"
                f"Le fichier ne contient que {len(cv_text.strip())} caractères "
                f"(minimum requis : {MIN_CV_LENGTH}).\n\n"
                "Veuillez soumettre un CV complet incluant : formation, "
                "expériences professionnelles et compétences."
            ),
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
# Prompt analyse
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH chez Sonatrach chargé d'évaluer objectivement des candidatures.

Analyse le CV ci-dessous par rapport au poste "{poste}"
en utilisant les exigences récupérées depuis la base documentaire interne.

=== EXIGENCES DU POSTE ===
{job_context}

=== CV DU CANDIDAT ===
{cv_text}

════════════════════════════════════════════════════════════════
ÉTAPE 1 — VÉRIFICATION DU DOMAINE (à faire en premier)
════════════════════════════════════════════════════════════════

Le domaine du candidat (formation + expérience principale) est-il compatible
avec le domaine du poste ?

Exemples de domaines INCOMPATIBLES (score = 0 obligatoire) :
  • Poste Finance/Comptabilité  ←→  Soudeur, Mécanicien, Forage, BTP, Géologie
  • Poste Technique/Ingénierie  ←→  Lettres, Sciences Humaines sans lien technique
  • Poste Juridique             ←→  Profil technique ou scientifique sans formation droit
  • Poste HSE                   ←→  Profil purement administratif sans formation sécurité

Exemples de domaines COMPATIBLES (noter normalement selon le barème) :
  • Poste Finance/Comptabilité  ←→  Diplôme finance, comptabilité, économie, gestion, audit
  • Poste Informatique/IT       ←→  Diplôme informatique, data, systèmes d'information
  • Poste HSE                   ←→  Formation HSE, sécurité industrielle, environnement
  • Poste RH                    ←→  Diplôme RH, droit du travail, sciences sociales

RÈGLE ABSOLUE :
→ Si domaines INCOMPATIBLES : score = 0/10 et passer directement au format de réponse
→ Si domaines COMPATIBLES : continuer avec le barème ci-dessous

════════════════════════════════════════════════════════════════
ÉTAPE 2 — NOTATION (uniquement si domaines compatibles)
════════════════════════════════════════════════════════════════

Évalue le candidat en te basant UNIQUEMENT sur ce qui est écrit dans le CV.
NE JAMAIS inventer, supposer ou inférer des compétences non mentionnées.

BARÈME :
  1-2  : Domaine correct mais profil quasi-inexistant (formation très éloignée, aucune expérience)
  3-4  : Faible adéquation (domaine correct, mais compétences ou expérience insuffisantes)
  5-6  : Adéquation moyenne (compétences partielles, expérience limitée)
  7-8  : Bonne adéquation (compétences solides, expérience significative)
  9-10 : Excellente adéquation (profil expert, expérience étendue, compétences complètes)

Règles de notation complémentaires :
  - Expérience directe dans le secteur pétrolier/gazier : +1 point bonus
  - Aucune expérience professionnelle dans le domaine : -2 points
  - Formations uniquement sans expérience : score maximum 6
  - CV vide ou quasi-vide : score 0-2 maximum

════════════════════════════════════════════════════════════════
ÉTAPE 3 — RÈGLES D'ANALYSE
════════════════════════════════════════════════════════════════

1. Baser l'analyse UNIQUEMENT sur le contenu explicite du CV
2. Si une information est absente : écrire "Non mentionné dans le CV"
3. Un lien GitHub/LinkedIn seul n'est pas une compétence
4. Les qualités personnelles seules ne justifient pas un bon score
5. Signaler clairement si le CV est incomplet

════════════════════════════════════════════════════════════════
FORMAT DE RÉPONSE OBLIGATOIRE
════════════════════════════════════════════════════════════════

**SCORE DE CORRESPONDANCE** : X/10

**DOMAINE** : Même domaine / Hors domaine

**POINTS FORTS**
- (lister uniquement ce qui est explicitement dans le CV)

**POINTS FAIBLES / MANQUANTS**
- (compétences ou expériences requises absentes du CV)

**RECOMMANDATION FINALE**
[Recommandé / À étudier / Non recommandé] — justification précise en 2-3 phrases.

**REMARQUES**
Observations complémentaires si nécessaire.

**ANNÉES_EXPÉRIENCE** : [entier, nombre d'années d'expérience dans le domaine du poste, -1 si inconnu]

**ANNÉE_DIPLOME** : [année du diplôme le plus élevé en lien avec le poste, 0 si absent ou inconnu]

**POSTE RECOMMANDÉ** : [poste Sonatrach le plus adapté au profil réel du candidat,
basé uniquement sur ce qui est écrit dans le CV.
Si CV insuffisant : "Indéterminable — CV insuffisant"]
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste Sonatrach",
        job_context=job_context or (
            "Aucun document spécifique trouvé dans la base. "
            "Appliquer les critères RH généraux Sonatrach pour ce type de poste."
        ),
        cv_text=cv_text[:4000],
    )


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(
    pipeline, cv_text: str, poste: str, filename: str = "CV"
) -> dict:
    import time
    t0 = time.time()

    # ── Validation préalable ───────────────────────────────────
    validation_error = validate_cv_text(cv_text, filename)
    if validation_error:
        logger.warning(
            "CV '%s' rejeté : contenu insuffisant (%d caractères)",
            filename, len(cv_text.strip())
        )
        return validation_error

    search_poste = poste.strip() if poste else ""

    if search_poste:
        search_query = f"exigences compétences diplômes requis poste {search_poste}"
    else:
        cv_hint = " ".join(cv_text[:600].split())[:300]
        search_query = f"poste Sonatrach requis diplôme expérience {cv_hint}"

    # ── Recherche RAG ──────────────────────────────────────────
    try:
        query_embedding = pipeline.embedder.embed_single(search_query)

        dense_results = pipeline.vector_store.search(
            query_embedding,
            k=pipeline.config.top_k_dense,
        )

        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(
                search_query,
                k=pipeline.config.top_k_sparse,
            )

        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=pipeline.config.rrf_k,
        )

        top_chunks = fused[:pipeline.config.top_k_after_rerank]

        if pipeline.reranker and top_chunks:
            pairs = [(search_query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)

            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)

            top_chunks = sorted(
                top_chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in top_chunks
        )

        sources = list({
            c["metadata"].get("source", "?")
            for c in top_chunks
        })

    except Exception as e:
        logger.warning("Erreur recherche RAG : %s", e)
        job_context = ""
        sources = []

    # ── Appel LLM ─────────────────────────────────────────────
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
    )

    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=(
                "Tu es un expert RH chez Sonatrach. Réponds uniquement en français. "
                "Base-toi UNIQUEMENT sur le contenu explicite du CV fourni, sans invention. "
                "Si le domaine du candidat est incompatible avec le poste, le score est 0/10. "
                "Si le domaine est compatible, évalue honnêtement selon le barème : "
                "un excellent profil dans le bon domaine doit obtenir 8 à 10/10."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )

    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score             = _extract_score(answer)
    recommended_poste = _extract_recommended_poste(answer)
    years_experience  = _extract_years_experience(answer)
    diploma_year      = _extract_diploma_year(answer)

    if score is None:
        logger.warning(
            "Score non extrait pour '%s'. Début réponse LLM :\n%s",
            filename, answer[:500]
        )

    elapsed = round(time.time() - t0, 2)

    return {
        "answer":            answer,
        "score":             score,
        "poste":             search_poste or recommended_poste or "Non précisé",
        "recommended_poste": recommended_poste or "Non précisé",
        "sources":           sources,
        "elapsed_seconds":   elapsed,
        "years_experience":  years_experience,
        "diploma_year":      diploma_year,
        "filename":          filename,
    }


# ─────────────────────────────────────────────────────────────
# Tri batch avec départage
# ─────────────────────────────────────────────────────────────

def sort_results_with_tiebreaker(results: list) -> list:
    """
    Trie les résultats batch par ordre décroissant :
      1. Score décroissant                    (critère principal)
      2. Années d'expérience décroissantes    (1er départage)
      3. Année du diplôme croissante          (2e départage : diplôme plus ancien = plus de recul)

    Utilisation dans le backend :
        sorted_results = sort_results_with_tiebreaker(analysis_results)
    """
    def sort_key(r: dict) -> tuple:
        score = r.get("score")
        if score is None:
            score = _extract_score(r.get("answer", ""))
        score = score if score is not None else -1

        years_exp = r.get("years_experience")
        years_exp = years_exp if (years_exp is not None and years_exp >= 0) else -1

        diploma_year = r.get("diploma_year")
        diploma_year = diploma_year if (diploma_year and diploma_year > 0) else 9999

        return (
            -score,        # décroissant
            -years_exp,    # décroissant
            diploma_year,  # croissant
        )

    return sorted(results, key=sort_key)


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
    import re

    patterns = [
        r"SCORE[^:\n]*:\s*\**\s*(\d{1,2})\s*\**\s*/\s*10",
        r"SCORE[^:\n]*:\s*\[?(\d{1,2})\]?\s*/\s*10",
        r"\*\*(\d{1,2})\*\*\s*/\s*10",
        r":\s*(\d{1,2})\s*/10",
        r"(\d{1,2})\s*/\s*10",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 10:
                return val

    # Fallback : premier X/10 dans le texte
    m = re.search(r'\b(\d{1,2})/10\b', text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val

    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
    import re
    patterns = [
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            if value and "[" not in value and len(value) > 3:
                return value
    return None


def _extract_years_experience(text: str) -> Optional[int]:
    """
    Format attendu dans la réponse LLM : **ANNÉES_EXPÉRIENCE** : 7
    """
    import re
    patterns = [
        r"ANN[EÉ]ES?[_\s]EXP[EÉ]RIENCE[^:\n]*:\s*\**\s*(-?\d{1,2})\s*\**",
        r"ANN[EÉ]ES?[_\s]EXP[^:\n]*:\s*(-?\d{1,2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if -1 <= val <= 50:
                return val
    return None


def _extract_diploma_year(text: str) -> Optional[int]:
    """
    Format attendu dans la réponse LLM : **ANNÉE_DIPLOME** : 2013
    """
    import re
    patterns = [
        r"ANN[EÉ]E[_\s]DIPLOM[EÉ][^:\n]*:\s*\**\s*((?:19|20)\d{2})\s*\**",
        r"ANN[EÉ]E[_\s]DIPLOM[^:\n]*:\s*((?:19|20)\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None
