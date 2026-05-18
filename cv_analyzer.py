"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.

Fixes v2:
- POSTE RECOMMANDÉ contraint à la liste GTP réelle (extraite du référentiel RAG)
- Score STRICTEMENT lié au poste demandé (pas au profil général)
- Double recherche RAG : exigences du poste + postes GTP compatibles avec le CV
- Validation post-LLM du titre recommandé contre la liste GTP
"""

import io
import logging
import re
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
# Extraction des titres de postes GTP depuis les chunks RAG
# ─────────────────────────────────────────────────────────────

def extract_gtp_post_titles(chunks: list) -> list[str]:
    """
    Extrait les titres de postes GTP valides depuis les chunks RAG.
    Cherche le pattern "Poste de base: XYZ" présent dans le référentiel.
    """
    titles = []
    seen = set()
    for chunk in chunks:
        content = chunk.get("content", "")
        matches = re.findall(r"Poste de base:\s*([^\n|]+)", content, re.IGNORECASE)
        for m in matches:
            title = m.strip().rstrip(".")
            if title and len(title) > 3 and title.lower() not in seen:
                seen.add(title.lower())
                titles.append(title)
    return titles


def find_closest_gtp_post(candidate: str, valid_titles: list[str]) -> Optional[str]:
    """
    Trouve le titre GTP le plus proche d'une chaîne proposée par le LLM.
    Utilise une correspondance simple par tokens.
    """
    if not candidate or not valid_titles:
        return None

    candidate_tokens = set(re.findall(r"\w+", candidate.lower()))

    best_title = None
    best_score = 0
    for title in valid_titles:
        title_tokens = set(re.findall(r"\w+", title.lower()))
        overlap = len(candidate_tokens & title_tokens)
        # Score normalisé pour éviter de favoriser les titres très longs
        score = overlap / max(len(candidate_tokens), 1)
        if score > best_score:
            best_score = score
            best_title = title

    # On accepte uniquement si la correspondance est raisonnable (≥ 1 token commun)
    return best_title if best_score > 0 else None


# ─────────────────────────────────────────────────────────────
# Prompt analyse
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH senior chez GTP (Groupe Travaux Pétroliers).
Ta mission : évaluer objectivement le CV ci-dessous pour le poste "{poste}".

═══ EXIGENCES DU POSTE (référentiel GTP) ═══
{job_context}

═══ CV DU CANDIDAT ═══
{cv_text}

═══ LISTE DES POSTES GTP VALIDES ═══
(Le POSTE RECOMMANDÉ doit être un titre EXACT de cette liste)
{valid_posts_list}

━━━ ÉTAPE 1 — INVENTAIRE OBJECTIF (obligatoire avant de scorer) ━━━
Avant de donner un score, réponds mentalement à ces questions :
  A) Le CV contient-il une ou plusieurs expériences professionnelles listées ? OUI / NON
  B) Ces expériences sont-elles dans le secteur pétrolier/industriel ? OUI / NON / SANS OBJET
  C) La formation est-elle en adéquation avec le poste ? OUI / PARTIELLE / NON
  D) Les compétences techniques requises pour "{poste}" sont-elles présentes ? OUI / PARTIELLE / NON

━━━ ÉTAPE 2 — BARÈME STRICT ━━━

Score de base selon l'expérience professionnelle (A) :
  • Aucune expérience professionnelle mentionnée dans le CV → score de base MAXIMUM = 5/10
  • Expérience présente mais hors domaine → score de base MAXIMUM = 6/10
  • Expérience dans un domaine adjacent → score de base MAXIMUM = 7/10
  • Expérience directement dans le domaine du poste → score de base MAXIMUM = 10/10

Modificateurs appliqués APRÈS le score de base :
  +1 si formation parfaitement alignée avec le poste (dans la limite du maximum)
  -1 si formation non technique pour un poste technique
  +1 si compétences techniques requises toutes présentes
  -1 si compétences clés du poste absentes du CV

RÈGLE ABSOLUE :
Un candidat sans AUCUNE expérience professionnelle ne peut jamais dépasser 5/10,
peu importe la qualité de sa formation ou de ses projets académiques.

━━━ FORMAT DE RÉPONSE OBLIGATOIRE (en français) ━━━

**INVENTAIRE**
- Expériences professionnelles : [OUI — liste / NON — aucune]
- Secteur pétrolier/industriel : [OUI / NON / Partiel]
- Formation adéquate : [OUI / PARTIELLE / NON]
- Compétences techniques requises : [OUI / PARTIELLE / NON]

**SCORE DE CORRESPONDANCE** : [chiffre 0-10]/10
(Justification du score en 1 ligne : score de base X + modificateurs)

**POINTS FORTS**
- [atouts du candidat par rapport au poste "{poste}"]

**POINTS FAIBLES / MANQUANTS**
- [lacunes par rapport aux exigences du poste "{poste}"]

**RECOMMANDATION FINALE**
[Recommandé / À étudier / Non recommandé] — justification courte.

**REMARQUES**
[Observations utiles pour le recruteur]

**POSTE RECOMMANDÉ** : [titre EXACT de la liste ci-dessus]
"""


def build_analysis_prompt(
    cv_text: str,
    poste: str,
    job_context: str,
    valid_gtp_posts: list[str],
) -> str:
    """Construit le prompt final avec la liste des postes GTP valides."""
    if valid_gtp_posts:
        posts_block = "\n".join(f"  • {p}" for p in valid_gtp_posts[:30])
    else:
        posts_block = "  (Aucun poste trouvé dans le référentiel — utilisez votre jugement RH)"

    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste GTP",
        job_context=job_context or (
            "Aucun document spécifique trouvé dans le référentiel. "
            "Appliquer les bonnes pratiques RH générales."
        ),
        cv_text=cv_text[:4000],
        valid_posts_list=posts_block,
    )


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str) -> dict:
    import time
    t0 = time.time()

    search_poste = poste.strip() if poste else ""

    # ── RECHERCHE 1 : exigences du poste demandé ──────────────────────────
    if search_poste:
        req_query = (
            f"exigences compétences diplômes formation requise poste {search_poste}"
        )
    else:
        cv_hint = " ".join(cv_text[:600].split())[:300]
        req_query = f"poste GTP requis diplôme expérience compétences {cv_hint}"

    # ── RECHERCHE 2 : postes GTP correspondant au profil du CV ───────────
    cv_summary = " ".join(cv_text[:800].split())[:400]
    profile_query = (
        f"Poste de base référentiel GTP compatible profil {cv_summary}"
    )

    job_context = ""
    sources = []
    valid_gtp_posts: list[str] = []

    try:
        from src.retrieval.hybrid_search import reciprocal_rank_fusion

        # — Recherche 1 : job requirements —
        req_emb = pipeline.embedder.embed_single(req_query)
        req_dense = pipeline.vector_store.search(req_emb, k=pipeline.config.top_k_dense)
        req_sparse = pipeline.bm25.search(req_query, k=pipeline.config.top_k_sparse) if pipeline.bm25 else []
        req_fused = reciprocal_rank_fusion(req_dense, req_sparse, k=pipeline.config.rrf_k)
        req_chunks = req_fused[:pipeline.config.top_k_after_rerank]

        if pipeline.reranker and req_chunks:
            pairs = [(req_query, c["content"]) for c in req_chunks]
            scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(req_chunks, scores):
                c["rerank_score"] = float(s)
            req_chunks = sorted(req_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in req_chunks
        )
        sources = list({c["metadata"].get("source", "?") for c in req_chunks})

        # — Recherche 2 : postes GTP valides pour ce profil —
        prof_emb = pipeline.embedder.embed_single(profile_query)
        prof_dense = pipeline.vector_store.search(prof_emb, k=10)
        prof_sparse = pipeline.bm25.search(profile_query, k=10) if pipeline.bm25 else []
        prof_fused = reciprocal_rank_fusion(prof_dense, prof_sparse, k=pipeline.config.rrf_k)
        prof_chunks = prof_fused[:15]

        # Combiner les chunks des deux recherches pour extraire les titres GTP
        all_chunks = req_chunks + prof_chunks
        valid_gtp_posts = extract_gtp_post_titles(all_chunks)

        logger.info("Postes GTP extraits pour recommandation: %d", len(valid_gtp_posts))

    except Exception as e:
        logger.warning("Erreur recherche RAG : %s", e)

    # ── Construction du prompt ────────────────────────────────────────────
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
        valid_gtp_posts=valid_gtp_posts,
    )

    # ── Appel LLM ─────────────────────────────────────────────────────────
    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=(
                "Tu es un expert RH chez GTP. "
                "Réponds UNIQUEMENT en français. "
                "Le POSTE RECOMMANDÉ doit être un titre EXACT de la liste fournie."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    # ── Parsing des résultats ─────────────────────────────────────────────
    score = _extract_score(answer)
    recommended_poste_raw = _extract_recommended_poste(answer)

    # Validation : le poste recommandé doit exister dans la liste GTP
    recommended_poste = _validate_recommended_poste(
        recommended_poste_raw, valid_gtp_posts
    )

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "Analyse terminée en %.2fs — score=%s — poste_rec=%s",
        elapsed, score, recommended_poste,
    )

    return {
        "answer": answer,
        "score": score,
        "poste": search_poste or recommended_poste or "Non précisé",
        "recommended_poste": recommended_poste or "Non précisé",
        "sources": sources,
        "elapsed_seconds": elapsed,
        "valid_gtp_posts_count": len(valid_gtp_posts),
    }


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
    patterns = [
        r"SCORE[^:\n]*:\s*\**\s*(\d{1,2})\s*\**\s*/\s*10",
        r"SCORE[^:\n]*:\s*\[?(\d{1,2})\]?\s*/\s*10",
        r"\*\*(\d{1,2})\*\*\s*/\s*10",
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
    patterns = [
        # Inline : **POSTE RECOMMANDÉ** : Ingénieur Forage
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        # Ligne suivante
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            if value and "[" not in value and len(value) > 3:
                return value
    return None


def _validate_recommended_poste(
    candidate: Optional[str],
    valid_titles: list[str],
) -> Optional[str]:
    """
    Vérifie que le poste recommandé par le LLM existe dans le référentiel GTP.
    Si non, cherche le titre le plus proche.
    Si la liste est vide, retourne le candidat tel quel (fallback).
    """
    if not candidate:
        return None

    if not valid_titles:
        # Pas de liste de référence disponible → on garde la réponse LLM
        return candidate

    # Correspondance exacte (insensible à la casse)
    candidate_lower = candidate.strip().lower()
    for title in valid_titles:
        if title.lower() == candidate_lower:
            return title  # Retourne la casse officielle du référentiel

    # Correspondance partielle : le LLM a peut-être abrégé le titre
    closest = find_closest_gtp_post(candidate, valid_titles)
    if closest:
        logger.info(
            "Poste LLM '%s' → corrigé en titre GTP '%s'",
            candidate, closest,
        )
        return closest

    # Aucun match — on garde quand même la réponse LLM plutôt que "Non précisé"
    logger.warning("Poste recommandé '%s' absent du référentiel GTP.", candidate)
    return candidate
