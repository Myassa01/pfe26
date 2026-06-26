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
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
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
            "answer": (
                f"⚠️ CV insuffisant ({filename}) — contenu trop limité.\n\n"
                f"Le fichier ne contient que {len(cv_text.strip())} caractères "
                f"(minimum requis : {MIN_CV_LENGTH}).\n\n"
                "Veuillez soumettre un CV complet incluant : formation, expériences et compétences."
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
# Prompt
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un système ATS (Applicant Tracking System) expert en recrutement chez Sonatrach.

════════════════════════════════════════
POSTE CIBLE
════════════════════════════════════════
{poste}

════════════════════════════════════════
RÉFÉRENTIEL DES POSTES SONATRACH
════════════════════════════════════════
{job_context}

════════════════════════════════════════
CV DU CANDIDAT
════════════════════════════════════════
{cv_text}

════════════════════════════════════════
ÉTAPE 1 — IDENTIFIER LE DOMAINE DU CV
════════════════════════════════════════

Lis attentivement le CV et identifie :
- Le domaine professionnel RÉEL du candidat (ex: Soudage, Comptabilité, Informatique…)
- Son diplôme le plus élevé et l'année
- Ses années d'expérience totales dans son domaine
- Ses compétences techniques principales

════════════════════════════════════════
ÉTAPE 2 — COMPARER AU POSTE CIBLE
════════════════════════════════════════

Si POSTE CIBLE est précisé :
  → Compare le domaine du CV avec le poste demandé.
  → RÈGLE ABSOLUE : un soudeur n'est PAS comptable. Un comptable n'est PAS développeur.
    Les domaines incompatibles = score 0 à 2/10, point final.
  → RÈGLE ABSOLUE : l'expérience Oil & Gas n'est un BONUS que si le candidat
    est déjà dans le bon domaine. Un soudeur chez Sonatrach qui postule comptable
    n'a AUCUN bonus — son expérience Oil & Gas est dans un autre métier.
  → Score basé UNIQUEMENT sur la pertinence du profil pour le poste demandé.

Si POSTE CIBLE est "Non précisé" :
  → N'invente PAS un poste au hasard.
  → Cherche dans le RÉFÉRENTIEL DES POSTES SONATRACH fourni ci-dessus
    le ou les postes qui correspondent AU DOMAINE RÉEL du candidat.
  → Si tu trouves un poste correspondant dans le référentiel, utilise son titre exact.
  → Si aucun poste dans le référentiel ne correspond, écris le titre de poste naturel
    qui correspond à son vrai métier.
  → Note le candidat sur sa capacité à occuper CE poste recommandé.

════════════════════════════════════════
BARÈME DE NOTATION (sur 10)
════════════════════════════════════════

CAS INCOMPATIBILITÉ TOTALE (domaines différents) :
→ Score = 0 à 2 maximum. Ne pas aller plus haut, même si le CV est excellent.

CAS COMPATIBILITÉ :

1. Diplôme/Formation (0–3 pts)
   3 = diplôme exactement requis pour ce poste
   2 = diplôme proche ou équivalent
   1 = formation partiellement liée
   0 = aucun diplôme pertinent

2. Compétences techniques spécifiques au poste (0–3 pts)
   3 = compétences clés du poste toutes présentes
   2 = compétences principales présentes, quelques lacunes
   1 = compétences partielles
   0 = compétences absentes

3. Expérience professionnelle dans le domaine du poste (0–3 pts)
   3 = expérience longue et directement pertinente (5+ ans)
   2 = expérience modérée et pertinente (2–5 ans)
   1 = expérience courte ou indirecte
   0 = aucune expérience pertinente

4. Expérience Sonatrach/Oil & Gas dans ce MÊME domaine (0–1 pt)
   1 = a travaillé chez Sonatrach/filiale dans le domaine du poste demandé
   0 = sinon (même si Oil & Gas dans un autre métier)

EXEMPLES CONCRETS :
- Nadia (comptable, 11 ans d'expérience, SAP, master finance) → poste "Comptable" → 8 ou 9/10
- Hamdi (soudeur, 19 ans) → poste "Comptable" → 0/10, Incompatible
- Hamdi (soudeur, 19 ans) → poste "Soudeur pipeline" → 9 ou 10/10
- Hamdi (soudeur, 19 ans) → poste "Chef d'équipe soudeurs" → Compatible, score selon expérience encadrement
- Rania (informaticienne) → poste "Chef d'équipe soudeurs" → 0 à 2/10, Incompatible

════════════════════════════════════════
POSTE RECOMMANDÉ — RÈGLES STRICTES
════════════════════════════════════════

Le POSTE RECOMMANDÉ doit être :
1. Basé sur le domaine RÉEL du CV (pas sur le poste demandé)
2. Choisi de préférence parmi les postes du RÉFÉRENTIEL SONATRACH fourni
3. Un seul titre précis — jamais une liste, jamais vague
4. Cohérent avec le niveau du candidat (BEP ≠ Ingénieur)

Exemples :
- Soudeur qualifié pipeline → "Soudeur Qualifié" ou titre exact du référentiel
- Comptable avec master → "Comptable Principal" ou titre exact du référentiel
- Développeur → "Développeur Informatique" ou titre exact du référentiel

════════════════════════════════════════
⚠️ FORMAT DE RÉPONSE — OBLIGATOIRE ET STRICT
════════════════════════════════════════

INTERDICTIONS :
- PAS de tableaux markdown (|col|col|)
- PAS de listes numérotées
- PAS de phrases de justification longues
- PAS d'introduction ni de conclusion
- PAS de texte hors des sections définies

FORMAT EXACT À RESPECTER :

**SCORE** : X/10
**DOMAINE** : Compatible / Partiellement compatible / Incompatible
**DÉCISION** : Recommandé / À étudier / Non recommandé

**ATOUTS**
- [point concis issu du CV, 1 ligne]
- [point concis issu du CV, 1 ligne]
- [point concis issu du CV, 1 ligne]

**LACUNES**
- [point concis, 1 ligne]
- [point concis, 1 ligne]

**POSTE RECOMMANDÉ** : [titre exact du poste, issu du référentiel si possible]

**ANNÉES_EXPÉRIENCE** : [nombre entier, ex: 11, ou -1 si inconnu]
**ANNÉE_DIPLOME** : [année ex: 2013, ou 0 si inconnue]
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    poste_label = poste if poste else "Non précisé"
    return ANALYSIS_PROMPT.format(
        poste=poste_label,
        job_context=job_context or (
            "Aucun référentiel disponible. "
            "Utilise les titres de postes courants dans l'industrie pétrolière algérienne."
        ),
        cv_text=cv_text[:4000],
    )


# ─────────────────────────────────────────────────────────────
# Post-processing : correction de score (anti-hallucination)
# ─────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "soudage":       ["soud", "pipeline", "smaw", "tig", "mig", "chaudronnerie", "wps", "asme"],
    "comptabilite":  ["compt", "fiscal", "bilan", "sage", "sap fi", "tva", "ibs", "irg", "audit"],
    "informatique":  ["data", "python", "sql", "développ", "logiciel", "cloud", "ml", "ia", "spark"],
    "rh":            ["ressources humaines", "recrutement", "paie", "rh", "gpec"],
    "juridique":     ["juriste", "droit", "contrat", "avocat", "juridique"],
    "géologie":      ["géolog", "forage", "réservoir", "sismique", "pétrophys"],
    "électricité":   ["électr", "haute tension", "basse tension", "automatisme", "instrumentation"],
    "mécanique":     ["mécanicien", "maintenance", "hydraulique", "pneumatique", "turbine"],
}


def detect_domain(text: str) -> str:
    """Retourne le domaine dominant du texte (CV ou poste)."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    if not any(scores.values()):
        return "inconnu"
    return max(scores, key=scores.get)


def postprocess_score(
    answer: str, score: Optional[int], cv_text: str, poste: str
) -> Optional[int]:
    """
    Corrige le score si le LLM a été trop généreux malgré une incompatibilité.

    Règles :
      1. Si le domaine CV et le domaine du poste sont clairement différents → score ≤ 2
         (détection objective par mots-clés, indépendante du texte LLM)
      2. Si le LLM dit INCOMPATIBLE *et* les domaines sont différents → confirme le plafond
         (on n'applique PAS le plafond sur "incompatible" seul : le LLM peut écrire
          "Incompatible avec le poste de chef d'équipe" pour un soudeur postulant chef soudeurs,
          ce qui est une confusion de niveau et non une incompatibilité de domaine)
    """
    if score is None:
        return score

    # Détection de domaine (source de vérité objective)
    domains_differ = False
    if poste:
        cv_domain    = detect_domain(cv_text)
        poste_domain = detect_domain(poste)
        if (
            cv_domain    != "inconnu"
            and poste_domain != "inconnu"
            and cv_domain    != poste_domain
        ):
            domains_differ = True

    # Règle unique : plafond à 2 uniquement si les domaines sont objectivement différents
    if domains_differ:
        return min(score, 2)

    return score


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(
    pipeline, cv_text: str, poste: str, filename: str = "CV"
) -> dict:
    import time
    t0 = time.time()

    validation_error = validate_cv_text(cv_text, filename)
    if validation_error:
        logger.warning(
            "CV '%s' rejeté : contenu insuffisant (%d car.)", filename, len(cv_text.strip())
        )
        return validation_error

    search_poste = poste.strip() if poste else ""

    # ── Requête RAG ────────────────────────────────────────────
    if search_poste:
        search_query = f"exigences diplômes compétences requis poste {search_poste} Sonatrach"
    else:
        cv_keywords  = " ".join(cv_text[:800].split())[:400]
        search_query = f"postes Sonatrach référentiel {cv_keywords}"

    try:
        query_embedding = pipeline.embedder.embed_single(search_query)
        dense_results   = pipeline.vector_store.search(query_embedding, k=pipeline.config.top_k_dense)

        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(search_query, k=pipeline.config.top_k_sparse)

        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused      = reciprocal_rank_fusion(dense_results, sparse_results, k=pipeline.config.rrf_k)
        top_chunks = fused[:pipeline.config.top_k_after_rerank]

        if pipeline.reranker and top_chunks:
            pairs  = [(search_query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)
            top_chunks = sorted(top_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}" for c in top_chunks
        )
        sources = list({c["metadata"].get("source", "?") for c in top_chunks})

    except Exception as e:
        logger.warning("Erreur RAG : %s", e)
        job_context = ""
        sources     = []

    # ── Appel LLM ─────────────────────────────────────────────
    prompt = build_analysis_prompt(cv_text=cv_text, poste=search_poste, job_context=job_context)

    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=(
                "Tu es un système ATS RH chez Sonatrach. Réponds UNIQUEMENT en français. "
                "RESPECTE STRICTEMENT le format demandé : **SCORE**, **DOMAINE**, **DÉCISION**, "
                "**ATOUTS**, **LACUNES**, **POSTE RECOMMANDÉ**, **ANNÉES_EXPÉRIENCE**, **ANNÉE_DIPLOME**. "
                "JAMAIS de tableaux markdown. JAMAIS de listes numérotées. "
                "Sois strict et objectif : un soudeur postulant comptable = 0/10 incompatible. "
                "Un comptable expérimenté pour un poste comptable = score élevé 7-9/10. "
                "Le bonus Oil&Gas ne s'applique QUE si l'expérience pétrolière est dans le MÊME domaine que le poste."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur LLM : {e}")

    # ── Extraction + post-processing ──────────────────────────
    score             = _extract_score(answer)
    score             = postprocess_score(answer, score, cv_text, search_poste)  # ← correction anti-hallucination
    recommended_poste = _extract_recommended_poste(answer)
    years_experience  = _extract_years_experience(answer)
    diploma_year      = _extract_diploma_year(answer)

    if score is None:
        logger.warning("Score non extrait pour '%s'. Réponse :\n%s", filename, answer[:500])

    elapsed         = round(time.time() - t0, 2)
    displayed_poste = search_poste or recommended_poste or "Non précisé"

    return {
        "answer":            answer,
        "score":             score,
        "poste":             displayed_poste,
        "recommended_poste": recommended_poste or "Non précisé",
        "sources":           sources,
        "elapsed_seconds":   elapsed,
        "years_experience":  years_experience,
        "diploma_year":      diploma_year,
        "filename":          filename,
    }


# ─────────────────────────────────────────────────────────────
# Tri batch
# ─────────────────────────────────────────────────────────────

def sort_results_with_tiebreaker(results: list) -> list:
    def sort_key(r: dict) -> tuple:
        score = r.get("score")
        if score is None:
            score = _extract_score(r.get("answer", ""))
        score = score if score is not None else -1

        years_exp    = r.get("years_experience")
        years_exp    = years_exp if (years_exp is not None and years_exp >= 0) else -1

        diploma_year = r.get("diploma_year")
        diploma_year = diploma_year if (diploma_year and diploma_year > 0) else 9999

        return (-score, -years_exp, diploma_year)

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
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*-][^\n\[\]\*]{3,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            if value and "[" not in value and len(value) > 3:
                return value
    return None


def _extract_years_experience(text: str) -> Optional[int]:
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
