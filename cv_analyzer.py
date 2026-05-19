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

MIN_CV_LENGTH = 200  # caractères minimum pour un CV exploitable


def validate_cv_text(cv_text: str, filename: str) -> Optional[dict]:
    """
    Vérifie que le CV contient suffisamment d'informations pour être analysé.
    Retourne un dict d'erreur si le CV est invalide, None sinon.
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
Tu es un expert RH chez Sonatrach chargé d'évaluer des candidatures.

Analyse le CV ci-dessous par rapport au poste "{poste}"
en utilisant les exigences récupérées depuis la base documentaire interne.

=== EXIGENCES DU POSTE ===
{job_context}

=== CV DU CANDIDAT ===
{cv_text}

=== RÈGLES ABSOLUES D'ANALYSE ===
1. Tu dois te baser UNIQUEMENT sur ce qui est EXPLICITEMENT écrit dans le CV ci-dessus.
2. NE JAMAIS inférer, supposer ou inventer des compétences non mentionnées.
3. Un lien GitHub ou LinkedIn N'EST PAS une compétence déclarée — ne pas en déduire des langages ou des projets.
4. Si une information est absente du CV, écrire explicitement "Non mentionné dans le CV".
5. Si le CV est très incomplet, le signaler clairement dans les remarques.

=== DÉTECTION HORS DOMAINE — RÈGLE PRIORITAIRE ABSOLUE ===
Avant toute notation, détermine si le domaine de formation et d'expérience du candidat
correspond au domaine du poste visé.

EXEMPLES DE DOMAINES INCOMPATIBLES (liste non exhaustive) :
- Poste Finance / Comptabilité  ←→  Profil Soudeur, Mécanicien, Électricien, Géologue, Forage, BTP
- Poste Technique / Ingénierie  ←→  Profil Lettres, Sciences Humaines, Commerce pur
- Poste HSE                     ←→  Profil purement administratif sans formation sécurité
- Poste Informatique / IT       ←→  Profil manuel ou artisanal sans compétences numériques
- Poste Juridique               ←→  Profil technique ou scientifique sans formation droit

RÈGLE HORS DOMAINE — NON NÉGOCIABLE :
→ Si les domaines sont incompatibles : Score OBLIGATOIREMENT 0/10
→ Aucune compétence transversale (communication, ponctualité, travail en équipe)
   ne peut justifier un score supérieur à 0 si le domaine est incompatible
→ Mentionner explicitement : "Profil hors domaine — aucune adéquation avec le poste visé"

=== BARÈME STRICT (uniquement si même domaine) ===
- 0    : Profil hors domaine (domaines incompatibles) — OBLIGATOIRE
- 1-2  : Même domaine, profil quasi-inexistant ou formation très éloignée
- 3-4  : Faible adéquation (domaine proche, compétences insuffisantes)
- 5-6  : Adéquation moyenne
- 7-8  : Bonne adéquation
- 9-10 : Excellente adéquation

=== RÈGLES DE NOTATION COMPLÉMENTAIRES ===
- CV vide ou quasi-vide => score 0 à 2 maximum
- Sans expérience dans le secteur pétrolier/gazier => pénalité de 1 à 2 points
- Qualités personnelles seules ≠ bon score
- Les formations courtes (< 1 semaine) ne compensent pas l'absence de diplôme adapté

=== CRITÈRES DE DÉPARTAGE (OBLIGATOIRES) ===
Pour permettre le classement en cas d'égalité de score, extraire :

ANNÉES_EXPÉRIENCE : nombre total d'années d'expérience professionnelle dans le domaine du poste.
- Calculé à partir des dates de début et de fin mentionnées dans le CV
- Année courante = 2025 (pour les postes "à présent" / "présent" / "actuellement")
- Si aucune expérience dans le domaine du poste : indiquer 0
- Si les dates sont absentes : indiquer -1

ANNÉE_DIPLOME : année d'obtention du diplôme le plus élevé EN LIEN avec le poste.
- Si plusieurs diplômes en lien avec le poste, prendre le plus récent
- Si aucun diplôme en lien avec le poste : indiquer 0
- Si l'année est absente : indiquer 0

=== FORMAT DE RÉPONSE OBLIGATOIRE ===
Réponds STRICTEMENT en français avec ce format exact (sans modifier les balises) :

**SCORE DE CORRESPONDANCE** : [0-10]/10

**DOMAINE** : [Même domaine / Hors domaine]

**POINTS FORTS**
- (uniquement ce qui est explicitement mentionné dans le CV)
- (écrire "Aucun point fort pertinent pour ce poste" si hors domaine)

**POINTS FAIBLES / MANQUANTS**
- (compétences ou expériences requises absentes du CV)

**RECOMMANDATION FINALE**
Recommandé / À étudier / Non recommandé — avec justification précise.

**REMARQUES**
Observations supplémentaires. Signaler si le CV est incomplet, hors domaine ou peu détaillé.

**ANNÉES_EXPÉRIENCE** : [nombre entier uniquement, ex: 7]

**ANNÉE_DIPLOME** : [année entière uniquement, ex: 2013]

**POSTE RECOMMANDÉ** : [intitulé du poste Sonatrach le PLUS ADAPTÉ au profil réel du candidat,
basé UNIQUEMENT sur sa formation et son expérience explicitement mentionnées dans le CV.
Si le CV est trop vide pour déterminer un poste, écrire "Indéterminable — CV insuffisant"]
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste Sonatrach",
        job_context=job_context or (
            "Aucun document spécifique trouvé. "
            "Utiliser bonnes pratiques RH générales."
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

    # ── Validation préalable du contenu du CV ──────────────────
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

    # ── Prompt final ───────────────────────────────────────────
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
                "Tu ne dois JAMAIS inventer ou supposer des informations absentes du CV fourni. "
                "Toute affirmation doit être directement justifiable par le texte du CV. "
                "Un profil dont le domaine est incompatible avec le poste reçoit "
                "OBLIGATOIREMENT 0/10, sans aucune exception possible."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )

    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score            = _extract_score(answer)
    recommended_poste = _extract_recommended_poste(answer)
    years_experience  = _extract_years_experience(answer)
    diploma_year      = _extract_diploma_year(answer)

    # Log debug si score non extrait
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
        "years_experience":  years_experience,   # critère départage 1
        "diploma_year":      diploma_year,        # critère départage 2
        "filename":          filename,
    }


# ─────────────────────────────────────────────────────────────
# Tri batch avec départage
# ─────────────────────────────────────────────────────────────

def sort_results_with_tiebreaker(results: list) -> list:
    """
    Trie les résultats par ordre décroissant selon :
      1. Score décroissant  (critère principal)
      2. Années d'expérience décroissantes  (1er départage : plus expérimenté gagne)
      3. Année du diplôme croissante  (2e départage : diplôme le plus ancien = plus de recul)

    Usage dans le backend batch :
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
            -years_exp,    # décroissant (plus d'exp = mieux classé)
            diploma_year,  # croissant   (diplôme plus ancien = plus de recul pro)
        )

    return sorted(results, key=sort_key)


# ─────────────────────────────────────────────────────────────
# Parsers internes
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

    # Fallback : premier X/10 trouvé n'importe où dans le texte
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
    Extrait le nombre d'années d'expérience depuis la réponse LLM.
    Format attendu dans la réponse : **ANNÉES_EXPÉRIENCE** : 7
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
    Extrait l'année du diplôme le plus élevé depuis la réponse LLM.
    Format attendu dans la réponse : **ANNÉE_DIPLOME** : 2013
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
