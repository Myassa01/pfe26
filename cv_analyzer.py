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
# Prompt analyse — règles structurelles, pas de liste de métiers
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH senior chez Sonatrach spécialisé dans le recrutement technique.

Ta mission :
Évaluer OBJECTIVEMENT un candidat pour le poste demandé.

Tu dois agir comme un vrai système ATS RH industriel :
- strict
- logique
- sans complaisance
- sans supposition
- sans invention
- JAMAIS hors du format de réponse imposé

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
ÉTAPE 1 — IDENTIFIER LES DEUX DOMAINES (OBLIGATOIRE, EN PREMIER)
════════════════════════════════════════

Avant toute notation, détermine pour toi-même (sans l'écrire dans la réponse
finale hors du champ DOMAINE) :

A. Le DOMAINE MÉTIER RÉEL du candidat, déduit UNIQUEMENT des expériences,
   diplômes et compétences explicitement listés dans le CV.
   (ex : soudage, jardinage/espaces verts, comptabilité, développement
   logiciel, réseaux, HSE, juridique, RH, génie civil, data, maintenance
   mécanique, médical, etc. — n'importe quel métier est possible, cette
   liste n'est qu'illustrative, ne t'y limite jamais.)

B. Le DOMAINE MÉTIER DU POSTE CIBLE, déduit du titre du poste et des
   exigences fournies.

Compare ensuite A et B selon la grille suivante :

- MÊME FAMILLE DE MÉTIER (les compétences/outils/connaissances de A sont
  directement utilisables pour B) → domaine COMPATIBLE
- FAMILLE PROCHE OU CONNEXE (certaines compétences transférables existent,
  mais le cœur de métier diffère) → domaine PARTIELLEMENT COMPATIBLE
- AUCUN POINT COMMUN TECHNIQUE OU FONCTIONNEL (les compétences de A
  n'apportent rien de concret pour exercer B ; un professionnel de A
  devrait être formé depuis zéro pour faire B) → domaine INCOMPATIBLE

⚠️ RÈGLE CLÉ : ne te laisse JAMAIS influencer par la motivation, les soft
skills, la rigueur, le sérieux ou les années d'expérience DANS LE DOMAINE A
pour juger la compatibilité avec B. Seules les compétences TECHNIQUES
transférables comptent. Plus d'expérience dans un domaine totalement
étranger au poste ne rapproche pas le candidat du poste — au contraire,
cela confirme que son métier réel est ailleurs.

════════════════════════════════════════
ÉTAPE 2 — RÈGLE D'INCOMPATIBILITÉ ABSOLUE (NON NÉGOCIABLE)
════════════════════════════════════════

SI le domaine est jugé INCOMPATIBLE à l'étape 1 :

→ Compatibilité domaine = 0 point (sur 3)
→ Diplôme/Formation = 0 point AUTOMATIQUEMENT, même si le candidat a un
  diplôme (un diplôme hors-domaine ne compte pas pour CE poste)
→ Compétences techniques = 0 point AUTOMATIQUEMENT, même si le CV liste
  beaucoup de compétences (elles sont hors-sujet pour ce poste)
→ Expérience professionnelle = 0 point AUTOMATIQUEMENT, même si le
  candidat a 10, 15 ou 20 ans d'expérience (c'est de l'expérience dans
  UN AUTRE métier, donc non pertinente ici)
→ Bonus Oil & Gas = 0 point (le bonus ne s'applique qu'à un profil déjà
  pertinent sur le plan technique)

→ SCORE FINAL = 0/10 obligatoirement
→ DOMAINE = Incompatible
→ DÉCISION = Non recommandé

Il est INTERDIT mathématiquement et logiquement de donner plus de 0/10
à un domaine jugé incompatible, quels que soient les autres éléments du
CV. Aucune compensation n'est possible. Ancienneté, diplôme, sérieux,
langues : RIEN ne rattrape une incompatibilité de domaine.

SI le domaine est jugé PARTIELLEMENT COMPATIBLE :
→ Compatibilité domaine = 1 point (sur 3) maximum
→ SCORE FINAL = 4/10 maximum, même avec un excellent diplôme ou une
  grande expérience dans le domaine connexe

SI le domaine est jugé COMPATIBLE :
→ Appliquer le barème complet ci-dessous normalement.

════════════════════════════════════════
RÈGLES RH COMPLÉMENTAIRES (s'appliquent uniquement si domaine COMPATIBLE)
════════════════════════════════════════

1. Évaluer UNIQUEMENT les informations explicitement présentes dans le CV.
2. Ne jamais inventer de compétences ni supposer une expérience.
3. Les langues, soft skills et qualités générales ne compensent JAMAIS
   l'absence de compétences techniques du poste.
4. Le POSTE CIBLE défini ci-dessus est la référence principale. Si les
   EXIGENCES DU POSTE semblent hors sujet par rapport au POSTE CIBLE,
   les ignorer et se baser uniquement sur l'intitulé du POSTE CIBLE.
5. Un diplôme seul sans expérience pertinente → score maximum = 6/10.
6. Une expérience réelle directement liée au poste est obligatoire pour
   un score élevé (7/10 ou plus).
7. Les stages dans le domaine exact du poste comptent comme expérience
   partielle (pas nulle). Un profil junior avec stage pertinent + diplôme
   pertinent peut atteindre 5 à 6/10. Ne jamais donner un score
   d'incompatibilité (0-1) à un profil junior dont le domaine est
   réellement compatible.
8. Expérience Oil & Gas / Sonatrach dans le domaine pertinent → bonus +1.

════════════════════════════════════════
MODE SANS POSTE CIBLE PRÉCIS
════════════════════════════════════════

Si le POSTE CIBLE indiqué plus haut est "non précisé" :
- Le domaine A (candidat) et le domaine B (poste) sont alors identiques
  par définition : évalue la qualité du profil DANS SON PROPRE métier.
- Le score reflète la solidité du profil dans son domaine réel (diplôme +
  expérience + compétences cohérentes entre elles), pas une comparaison
  à un poste externe.
- Le POSTE RECOMMANDÉ doit être le métier réel et précis du candidat,
  déduit du CV.
- Le format de réponse imposé reste IDENTIQUE et obligatoire.

════════════════════════════════════════
BARÈME OBLIGATOIRE (si domaine compatible ou partiellement compatible)
════════════════════════════════════════

1. Compatibilité domaine (0 à 3 points)
   3 = domaine parfaitement adapté
   2 = domaine proche
   1 = domaine partiellement lié
   0 = domaine incompatible (→ voir ÉTAPE 2, score final forcé à 0)

2. Diplôme / Formation (0 à 2 points)
   2 = diplôme directement lié au poste
   1 = formation partiellement liée
   0 = aucun diplôme pertinent

3. Compétences techniques (0 à 2 points)
   2 = compétences solides et pertinentes
   1 = compétences partielles
   0 = compétences absentes

4. Expérience professionnelle (0 à 2 points)
   2 = expérience forte et pertinente
   1 = expérience limitée
   0 = aucune expérience pertinente

5. Bonus secteur Oil & Gas (0 à 1 point)
   1 = expérience pétrole/gaz/Sonatrach dans le domaine pertinent
   0 = aucune

════════════════════════════════════════
INTERPRÉTATION DU SCORE
════════════════════════════════════════

0   : Domaine incompatible (règle absolue, voir ÉTAPE 2)
1-2 : Très faible adéquation / domaine partiellement compatible faible
3-4 : Faible à moyenne adéquation
5-6 : Adéquation moyenne
7-8 : Bonne adéquation
9-10: Excellente adéquation

════════════════════════════════════════
RÈGLES POUR LE POSTE RECOMMANDÉ
════════════════════════════════════════

Le POSTE RECOMMANDÉ doit toujours refléter le VRAI métier du candidat,
déduit de son CV — jamais le métier du poste cible si le domaine est
incompatible.

Si COMPATIBLE avec le poste cible :
→ écrire le POSTE CIBLE (ou un titre très proche)

Si INCOMPATIBLE avec le poste cible :
→ proposer le titre de poste correspondant au vrai métier déduit du CV
→ ou écrire "Non pertinent pour ce poste" si aucun intitulé clair ne
  se dégage

⚠️ Un seul poste recommandé, jamais une liste. Une ligne, un titre précis,
cohérent avec les diplômes/expériences réels du CV (jamais le métier du
poste cible si le domaine a été jugé incompatible).

════════════════════════════════════════
FORMAT DE RÉPONSE OBLIGATOIRE — AUCUNE EXCEPTION
════════════════════════════════════════

Ce format doit être respecté EXACTEMENT, dans CE cas :
- poste précisé ou non précisé
- domaine compatible ou incompatible
- score élevé ou nul

Ne jamais répondre en texte libre. Ne jamais omettre une section. Ne
jamais ajouter de section supplémentaire. Ne jamais reformuler les
libellés des champs.

**SCORE** : X/10
**DOMAINE** : Compatible / Partiellement compatible / Incompatible
**DÉCISION** : Recommandé / À étudier / Non recommandé

**ATOUTS**
- point 1
- point 2
- point 3

**LACUNES**
- point 1
- point 2
- point 3

**POSTE RECOMMANDÉ** : [titre du poste, une seule ligne, pas de tiret, pas de liste]

**ANNÉES_EXPÉRIENCE** : entier ou -1
**ANNÉE_DIPLOME** : année ou 0

════════════════════════════════════════
IMPORTANT
════════════════════════════════════════

Ne jamais être "gentil". Être STRICT comme un vrai recruteur Sonatrach.

Un candidat dont le domaine est incompatible avec le poste cible ne doit
JAMAIS recevoir un score supérieur à 0, quelle que soit la qualité de
son CV dans son propre métier.

Respecte le format de réponse à la lettre, systématiquement.
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    poste_label = poste.strip() if poste and poste.strip() else (
        "Non précisé. Applique le MODE SANS POSTE CIBLE PRÉCIS décrit "
        "plus bas : identifie le domaine réel du candidat et évalue son "
        "profil dans son propre métier, en respectant strictement le "
        "format de réponse obligatoire."
    )
    return ANALYSIS_PROMPT.format(
        poste=poste_label,
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
        search_query = None  # pas de poste → pas de recherche RAG

    # ── Recherche RAG ──────────────────────────────────────────
    job_context = ""
    sources = []

    if search_query:
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
                "Sois CONCIS et DIRECT : le recruteur doit pouvoir lire la fiche en 20 secondes. "
                "Utilise EXCLUSIVEMENT le format imposé dans le prompt, avec exactement les "
                "mêmes libellés de champs (SCORE, DOMAINE, DÉCISION, ATOUTS, LACUNES, "
                "POSTE RECOMMANDÉ, ANNÉES_EXPÉRIENCE, ANNÉE_DIPLOME). "
                "N'écris JAMAIS de réponse en texte libre, de paragraphe d'introduction, "
                "ou de section non prévue par le format. Aucune phrase de remplissage. "
                "Base-toi UNIQUEMENT sur le contenu explicite du CV, sans invention. "
                "Si le domaine du candidat est incompatible avec le poste cible, le score "
                "DOIT être 0/10, sans aucune exception ni compensation. "
                "Le POSTE CIBLE indiqué dans le prompt est la référence principale ; si les "
                "exigences du poste mentionnent d'autres intitulés, ignore-les et évalue "
                "le CV par rapport au POSTE CIBLE uniquement."
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
      3. Année du diplôme croissante          (2e départage)
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
            -score,
            -years_exp,
            diploma_year,
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
            value = m.group(1).strip().strip("*•[] \t-")
            # prendre seulement la première ligne (ignorer les listes)
            value = value.split("\n")[0].strip().strip("*•[] \t-")
            if value and len(value) > 3:
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
