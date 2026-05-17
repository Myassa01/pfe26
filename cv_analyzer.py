"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
Les postes recommandés sont récupérés DYNAMIQUEMENT depuis le référentiel GTP (PDF)
via le pipeline RAG — aucune liste hardcodée.

CORRECTIONS APPLIQUÉES :
  - Anti-hallucination : le LLM ne peut plus inventer de compétences absentes du CV
  - cv_text étendu à 6000 chars (évite la troncature)
  - System prompt renforcé pour l'analyse individuelle et le classement
  - Règle stricte : profil hors domaine = score 0-2/10 sans exception
  - temperature=0.0 explicitement forcé partout
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
    """Dispatcher extraction selon le format du fichier."""
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
# Récupération des postes GTP depuis le RAG
# ─────────────────────────────────────────────────────────────

def retrieve_postes_gtp(pipeline) -> str:
    """
    Interroge le pipeline RAG pour récupérer la liste des postes GTP
    directement depuis le référentiel PDF indexé.
    Retourne une chaîne de contexte prête à être injectée dans le prompt.
    """
    query = "liste complète des postes GTP référentiel intitulés emplois"
    try:
        query_embedding = pipeline.embedder.embed_single(query)

        dense_results = pipeline.vector_store.search(
            query_embedding,
            k=pipeline.config.top_k_dense,
        )

        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(
                query,
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
            pairs = [(query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)
            top_chunks = sorted(
                top_chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )

        return "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in top_chunks
        ) or ""

    except Exception as e:
        logger.warning("Erreur récupération postes GTP via RAG : %s", e)
        return ""


# ─────────────────────────────────────────────────────────────
# Prompt analyse INDIVIDUELLE
# FIX : ajout règles anti-hallucination + hors domaine strict
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH senior chez GTP (Groupe Travaux Pétroliers), spécialisé dans
l'évaluation et le classement de profils techniques et ingénierie dans le secteur
pétrolier et de la construction industrielle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES FONDAMENTALES — À RESPECTER ABSOLUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 INTERDIT — Ne JAMAIS :
  - Attribuer au candidat des compétences NON mentionnées dans son CV
  - Inventer des formations, diplômes ou expériences absents du CV fourni
  - Supposer qu'un candidat maîtrise un outil ou une norme parce que le poste l'exige
  - Confondre les exigences du POSTE avec les compétences réelles du CANDIDAT
  - Copier les exigences du poste dans la section "Compétences Techniques" du candidat

✅ OBLIGATOIRE :
  - Analyser UNIQUEMENT ce qui est écrit textuellement dans le CV fourni
  - Si une compétence requise est absente du CV → la signaler comme MANQUANTE
  - Si le domaine du CV est totalement différent du poste → score 0-2/10, niveau HORS DOMAINE
  - Distinguer clairement : "le poste exige X" ≠ "le candidat possède X"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Poste cible évalué : "{poste}"

Tu dois :
1. Analyser ce CV par rapport au poste "{poste}"
2. Donner un score de correspondance STRICT selon le barème ci-dessous
3. Identifier le poste GTP officiel qui correspond le mieux au profil RÉEL du candidat
4. Si le poste du CV est SUPÉRIEUR au poste cible : expliquer pourquoi ce profil
   reste pertinent mais surdimensionné
5. Si le poste du CV est INFÉRIEUR au poste cible : expliquer le manque
6. Si le domaine est totalement différent : classer HORS DOMAINE, score 0-2/10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXIGENCES DU POSTE (depuis référentiel GTP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{job_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CV DU CANDIDAT — SOURCE UNIQUE D'INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Analyse UNIQUEMENT le texte ci-dessous. Ne rien ajouter, ne rien supposer.

{cv_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LISTE DES POSTES GTP OFFICIELS (extraite du référentiel PDF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{postes_gtp_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BARÈME DE SCORING STRICT (0-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORRESPONDANCE EXACTE (poste = poste cible) :
  10/10 → Profil idéal, toutes exigences remplies, expérience optimale
   9/10 → Profil excellent, 1-2 lacunes mineures
   8/10 → Très bonne adéquation, lacunes surmontables rapidement
   7/10 → Bonne adéquation, formation OK, expérience légèrement insuffisante

PROFIL SUPÉRIEUR (poste du CV > poste cible) :
   7/10 → Surdimensionné mais mobilisable, risque de désengagement
   6/10 → Trop surdimensionné, inadapté sauf besoin de montée en charge rapide

PROFIL INFÉRIEUR (poste du CV < poste cible) :
   5/10 → Légèrement en dessous, potentiel évolutif fort
   4/10 → En dessous des exigences, manques importants
   3/10 → Profil clairement insuffisant pour le poste

HORS DOMAINE :
   0-2/10 → Profil totalement inadapté, domaine sans lien

RÈGLES ABSOLUES :
✗ La motivation ou qualités personnelles seules ne font JAMAIS monter le score
✗ Diplôme non pertinent pour le domaine = score max 3/10
✗ Zéro expérience dans le domaine du poste = score max 3/10
✗ Domaine totalement différent (ex: soudeur pour poste Finance) = score 0-2/10 OBLIGATOIRE
✗ Un profil surdimensionné NE PEUT PAS dépasser 7/10 pour un poste inférieur
✗ NE PAS attribuer les exigences du poste comme compétences du candidat
✓ Expérience GTP ou Sonatrach directe dans le MÊME domaine = bonus +1 (max 10)
✓ Certifications spécifiques au poste = bonus +0.5 par certification pertinente

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE RÉPONSE OBLIGATOIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Réponds STRICTEMENT en français avec ce format exact, sans déviation :

**SCORE DE CORRESPONDANCE** : [0-10]/10

**NIVEAU DU PROFIL PAR RAPPORT AU POSTE CIBLE**
[EXACT / SUPÉRIEUR / INFÉRIEUR / HORS DOMAINE] — justification en 1 phrase

**POINTS FORTS** (uniquement ce qui est présent dans le CV)
- [point fort 1, lié directement aux exigences du poste ET présent dans le CV]
- [point fort 2]
- [point fort 3 minimum, ou "Aucun point fort pertinent pour ce poste" si hors domaine]

**POINTS FAIBLES / MANQUANTS** (compétences requises absentes du CV)
- [compétence requise par le poste mais ABSENTE du CV du candidat]
- [manque 2]

**ADÉQUATION HIÉRARCHIQUE**
Poste actuel/équivalent du candidat : [poste GTP le plus proche du profil RÉEL, issu du référentiel]
Poste cible demandé              : {poste}
Écart                            : [Exact / +N niveau(x) au-dessus / -N niveau(x) en dessous / Hors domaine]

**RECOMMANDATION FINALE**
[Recommandé / À étudier / Non recommandé] — justification courte et concrète.

**REMARQUES**
[Observations RH supplémentaires : risque de sur-qualification, potentiel d'évolution,
points à vérifier en entretien, etc.]

**POSTE RECOMMANDÉ** : [EXACTEMENT un intitulé de poste issu du référentiel GTP fourni ci-dessus — aucune invention]
"""


# ─────────────────────────────────────────────────────────────
# System prompt analyse individuelle — FIX anti-hallucination
# ─────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = (
    "Tu es un expert RH senior chez GTP. Réponds uniquement en français. "
    "RÈGLE N°1 ABSOLUE : Tu analyses UNIQUEMENT les informations présentes dans le CV fourni. "
    "Tu ne dois JAMAIS inventer, supposer ou inférer des compétences non mentionnées explicitement dans le CV. "
    "RÈGLE N°2 : Les exigences du poste décrivent ce que le poste REQUIERT, "
    "pas ce que le candidat POSSÈDE. Ne pas confondre les deux. "
    "RÈGLE N°3 : Un candidat dont le domaine est totalement différent du poste cible "
    "(ex: soudeur évalué pour un poste Finance) DOIT recevoir un score de 0 à 2/10 "
    "et le niveau HORS DOMAINE, sans exception. "
    "RÈGLE N°4 : Le POSTE RECOMMANDÉ doit être EXACTEMENT un intitulé issu du référentiel "
    "GTP fourni dans le contexte. Ne jamais inventer un poste."
)


# ─────────────────────────────────────────────────────────────
# Prompt CLASSEMENT multi-CVs
# ─────────────────────────────────────────────────────────────

RANKING_PROMPT = """
Tu es un expert RH senior chez GTP (Groupe Travaux Pétroliers).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES FONDAMENTALES — À RESPECTER ABSOLUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 INTERDIT :
  - Modifier les scores individuels déjà attribués
  - Attribuer des compétences non mentionnées dans les analyses individuelles
  - Classer un profil HORS DOMAINE autrement qu'en dernière position ou éliminé
  - Ignorer le niveau de profil lors du classement (EXACT > INFÉRIEUR > SUPÉRIEUR)

✅ OBLIGATOIRE :
  - Utiliser UNIQUEMENT les informations des analyses individuelles fournies ci-dessous
  - Appliquer strictement la priorité de classement hiérarchique
  - Un profil HORS DOMAINE est toujours classé en dernier et marqué "Non recommandé"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION : CLASSEMENT DE CANDIDATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Poste à pourvoir : "{poste}"

Tu as analysé {n_cvs} candidats. Voici les résultats individuels :
{analyses_individuelles}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES DE CLASSEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITÉ DE CLASSEMENT (dans l'ordre strict) :
1. EXACT        : profil dont le niveau correspond exactement au poste → classé EN PREMIER
2. INFÉRIEUR -1 : légèrement en dessous (-1 niveau), fort potentiel d'évolution → 2ème
3. SUPÉRIEUR +1 : légèrement surdimensionné (+1 niveau), mobilisable → 3ème
4. INFÉRIEUR -2 : deux niveaux en dessous → après
5. SUPÉRIEUR +2 : deux niveaux ou plus au-dessus (sur-qualification forte) → en dernier
6. HORS DOMAINE : éliminé du classement, mention explicite, toujours en dernière position

À SCORE ÉGAL, départage par ordre de priorité :
  a) Expérience directe GTP ou Sonatrach dans le MÊME domaine (bonus)
  b) Nombre d'années d'expérience dans le domaine exact du poste
  c) Certifications pertinentes et valides pour le poste
  d) Niveau de diplôme le plus adapté au poste

RAPPEL IMPORTANT :
- Un profil SUPÉRIEUR (ex: Ingénieur Principal pour poste Ingénieur) est classé APRÈS
  un profil EXACT, même si son score brut est identique.
- Un profil INFÉRIEUR avec fort potentiel d'évolution est préféré à un profil
  SUPÉRIEUR surdimensionné.
- Un profil HORS DOMAINE est TOUJOURS classé en dernier, quelle que soit sa qualité générale.
- Ne jamais remonter un profil HORS DOMAINE même s'il a des qualités générales.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE RÉPONSE OBLIGATOIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Réponds STRICTEMENT en français avec ce format exact :

**CLASSEMENT FINAL — POSTE : {poste}**

| Rang | Candidat | Score | Niveau profil     | Justification courte            |
|------|----------|-------|-------------------|---------------------------------|
|  1   | [Nom]    | [X/10]| [EXACT/SUP/INF/HORS DOMAINE] | [1 phrase max]     |
|  2   | [Nom]    | [X/10]| [EXACT/SUP/INF/HORS DOMAINE] | [1 phrase max]     |
| ...  | ...      | ...   | ...               | ...                             |

**CANDIDAT RECOMMANDÉ EN PRIORITÉ**
[Nom — Score/10 — Justification 2 phrases max expliquant pourquoi ce profil est le meilleur
choix pour le poste "{poste}"]

**CANDIDATS À ÉTUDIER**
- [Nom] : [raison courte — potentiel, lacunes comblables, etc.]
- ... (ou "Aucun" si tous sont recommandés ou non recommandés)

**CANDIDATS NON RECOMMANDÉS**
- [Nom] : [raison courte — hors domaine, trop surdimensionné, manques rédhibitoires]
- ... (ou "Aucun" si tous les candidats sont recommandés)

**NOTE RH GLOBALE**
[Observation sur le vivier : qualité globale des candidats, adéquation au poste,
recommandation si aucun profil n'est idéal (ex: élargir la recherche, former en interne)]
"""


# ─────────────────────────────────────────────────────────────
# System prompt classement — FIX anti-confusion profils
# ─────────────────────────────────────────────────────────────

RANKING_SYSTEM_PROMPT = (
    "Tu es un expert RH senior chez GTP. Réponds uniquement en français. "
    "RÈGLE N°1 : Applique strictement les règles de classement hiérarchique fournies. "
    "Un profil EXACT prime toujours sur un profil SUPÉRIEUR ou INFÉRIEUR, même à score égal. "
    "RÈGLE N°2 : Un profil HORS DOMAINE est TOUJOURS classé en dernière position "
    "et marqué 'Non recommandé', sans exception, même s'il a un score individuel élevé. "
    "RÈGLE N°3 : Ne pas modifier les scores individuels déjà attribués. "
    "RÈGLE N°4 : Utilise UNIQUEMENT les informations des analyses individuelles fournies. "
    "Ne pas inventer ni attribuer de nouvelles compétences aux candidats."
)


# ─────────────────────────────────────────────────────────────
# Builders de prompts
# ─────────────────────────────────────────────────────────────

def build_analysis_prompt(cv_text: str, poste: str, job_context: str,
                          postes_gtp_context: str) -> str:
    """Construit le prompt d'analyse individuelle."""
    # FIX : cv_text étendu à 6000 chars pour éviter troncature
    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste GTP",
        job_context=job_context or (
            "Aucun document spécifique trouvé. "
            "Utiliser le barème strict et le bon sens RH."
        ),
        cv_text=cv_text[:6000],  # FIX : était 4000, augmenté à 6000
        postes_gtp_context=postes_gtp_context or (
            "Référentiel des postes GTP non disponible. "
            "Utiliser les intitulés standards du secteur pétrolier algérien."
        ),
    )


def build_ranking_prompt(poste: str, analyses: list[dict]) -> str:
    """
    Construit le prompt de classement à partir des analyses individuelles déjà effectuées.

    :param poste: Intitulé du poste cible.
    :param analyses: Liste de dicts avec clés 'nom', 'answer', 'score', 'niveau_profil'.
    """
    blocs = []
    for i, a in enumerate(analyses, start=1):
        nom = a.get("nom", f"Candidat {i}")
        score = a.get("score", "N/A")
        niveau = a.get("niveau_profil", "Non déterminé")
        answer = a.get("answer", "")
        blocs.append(
            f"--- CANDIDAT {i} : {nom} ---\n"
            f"Score individuel : {score}/10\n"
            f"Niveau profil    : {niveau}\n"
            f"Analyse détaillée :\n{answer}\n"
        )

    return RANKING_PROMPT.format(
        poste=poste or "poste généraliste GTP",
        n_cvs=len(analyses),
        analyses_individuelles="\n".join(blocs),
    )


# ─────────────────────────────────────────────────────────────
# Analyse principale (individuelle)
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str) -> dict:
    import time
    t0 = time.time()

    search_poste = poste.strip() if poste else ""

    # ── 1. Recherche RAG : exigences du poste ─────────────────
    if search_poste:
        search_query = f"exigences compétences diplômes requis poste {search_poste}"
    else:
        cv_hint = " ".join(cv_text[:600].split())[:300]
        search_query = f"poste GTP requis diplôme expérience {cv_hint}"

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
        logger.warning("Erreur recherche RAG (exigences poste) : %s", e)
        job_context = ""
        sources = []

    # ── 2. Récupération des postes GTP depuis le référentiel PDF ─
    postes_gtp_context = retrieve_postes_gtp(pipeline)

    # ── 3. Construction du prompt ─────────────────────────────────
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
        postes_gtp_context=postes_gtp_context,
    )

    # ── 4. Appel LLM — FIX : system prompt renforcé + temperature=0.0 forcé ──
    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=ANALYSIS_SYSTEM_PROMPT,  # FIX : system prompt anti-hallucination
            temperature=0.0,                # FIX : explicitement forcé à 0.0
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score = _extract_score(answer)
    niveau_profil = _extract_niveau_profil(answer)
    recommended_poste = _extract_recommended_poste(answer)

    elapsed = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "score": score,
        "poste": search_poste or recommended_poste or "Non précisé",
        "recommended_poste": recommended_poste or "Non précisé",
        "niveau_profil": niveau_profil or "Non déterminé",
        "sources": sources,
        "elapsed_seconds": elapsed,
    }


# ─────────────────────────────────────────────────────────────
# Classement multi-CVs
# ─────────────────────────────────────────────────────────────

def rank_cvs_with_pipeline(pipeline, analyses: list[dict], poste: str) -> dict:
    """
    Classement de plusieurs CVs déjà analysés individuellement.

    :param pipeline: Pipeline RAG.
    :param analyses: Liste de dicts retournés par analyze_cv_with_pipeline(),
                     enrichis d'une clé 'nom' (nom du candidat / nom du fichier).
    :param poste: Poste cible pour le classement.
    :return: Dict avec 'answer' (texte classement) et métadonnées.
    """
    import time
    t0 = time.time()

    prompt = build_ranking_prompt(poste=poste, analyses=analyses)

    # FIX : system prompt renforcé + temperature=0.0 explicitement forcé
    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=RANKING_SYSTEM_PROMPT,   # FIX : system prompt anti-confusion
            temperature=0.0,                # FIX : explicitement forcé à 0.0
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur classement LLM : {e}")

    elapsed = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "poste": poste,
        "n_cvs": len(analyses),
        "elapsed_seconds": elapsed,
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


def _extract_niveau_profil(text: str) -> Optional[str]:
    """
    Extrait le niveau du profil par rapport au poste cible.
    Retourne : EXACT / SUPÉRIEUR / INFÉRIEUR / HORS DOMAINE
    """
    patterns = [
        r"NIVEAU DU PROFIL[^:\n]*:\s*\n?\s*\*?\*?(EXACT|SUP[EÉ]RIEUR|INF[EÉ]RIEUR|HORS DOMAINE)\*?\*?",
        r"\*?\*?(EXACT|SUP[EÉ]RIEUR|INF[EÉ]RIEUR|HORS DOMAINE)\*?\*?\s*[—–-]",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            raw = m.group(1).upper().strip()
            if "EXACT" in raw:
                return "EXACT"
            elif "SUP" in raw:
                return "SUPÉRIEUR"
            elif "INF" in raw:
                return "INFÉRIEUR"
            elif "HORS" in raw:
                return "HORS DOMAINE"
    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
    """Extrait le poste recommandé depuis la réponse du LLM."""
    patterns = [
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]{4,})",
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]{4,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            if value and "[" not in value and len(value) > 3:
                return value
    return None
