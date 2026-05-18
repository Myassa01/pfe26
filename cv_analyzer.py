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

═══ EXIGENCES DU POSTE VISÉ (référentiel GTP) ═══
{job_context}

═══ CV DU CANDIDAT ═══
{cv_text}

═══ LISTE DES POSTES GTP VALIDES ═══
{valid_posts_list}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 0 — LECTURE OBLIGATOIRE DU CV (NE PAS SAUTER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Avant tout scoring, lis INTÉGRALEMENT le CV ci-dessus et extrais :

  A) DOMAINE RÉEL : Quel est le domaine professionnel principal du candidat ?
     → Identifie-le à partir de sa FORMATION et de ses EXPÉRIENCES, pas du poste visé.
     → Exemples : comptabilité/finance, soudage/chaudronnerie, informatique, RH, géologie...

  B) ANNÉES D'EXPÉRIENCE : Calcule le total en années à partir des dates.
     → Si les dates se chevauchent, ne les cumule pas.
     → Date de référence : année actuelle = 2025.

  C) NIVEAU DE FORMATION : BEP/CAP → Bac → Licence → Master → Ingénieur+

  D) COMPÉTENCES TECHNIQUES : liste exacte issue du CV (ne pas inventer).

  E) CERTIFICATIONS : liste exacte ou "Aucune".

  F) ANNÉE DU DIPLÔME LE PLUS RÉCENT : pour départager les égalités.

  G) A-t-il une expérience de GESTION D'ÉQUIPE ? OUI (précise combien de personnes)
     ou NON. Ne pas inférer — doit être explicitement mentionné dans le CV.

RÉSUMÉ OBLIGATOIRE (complète avant de continuer) :
  - Domaine réel du CV       : ___
  - Années d'expérience      : ___
  - Niveau de formation      : ___
  - Gestion d'équipe         : OUI / NON
  - Certifications           : ___
  - Année dernier diplôme    : ___

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 1 — DÉTECTION HORS DOMAINE (PRIORITÉ ABSOLUE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare le DOMAINE RÉEL du CV (extrait à l'étape 0) avec le domaine du poste "{poste}".

RÈGLE : Le domaine est déterminé par le MÉTIER, pas par le secteur d'activité.
  → Un comptable qui a travaillé dans une entreprise pétrolière reste un COMPTABLE.
     Il est HORS DOMAINE pour tout poste technique (soudeur, ingénieur forage...).
  → Un soudeur est HORS DOMAINE pour tout poste administratif, finance, RH.
  → L'entreprise employeur ne définit PAS le domaine du candidat.

Exemples de correspondances HORS DOMAINE (SCORE = 0/10) :
  • Poste : soudeur / Chef d'équipe soudeurs → CV : comptable, financier, juriste, RH
  • Poste : ingénieur génie civil → CV : biologiste, enseignant, banquier
  • Poste : développeur informatique → CV : mécanicien, cuisinier, agent de sécurité
  • Poste : chef de chantier pétrolier → CV : pharmacien, architecte d'intérieur

→ Si HORS DOMAINE : SCORE = 0/10 (non négociable). Va directement au FORMAT DE RÉPONSE.
→ Si dans le domaine (même partiellement) : continue vers l'ÉTAPE 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 2 — NIVEAU DU CANDIDAT vs NIVEAU DU POSTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare le niveau RÉEL du candidat avec le niveau du poste "{poste}" :

Niveaux de postes (du plus bas au plus élevé) :
  DÉBUTANT    : 0-2 ans, aucune gestion d'équipe
  CONFIRMÉ    : 3-7 ans, compétences validées, éventuellement encadrement limité
  SENIOR      : 8-14 ans, expertise reconnue, peut encadrer
  CHEF D'ÉQUIPE : 8+ ans avec gestion d'équipe explicite dans le CV
  CHEF DE DÉPARTEMENT / MANAGER : 15+ ans avec gestion prouvée d'équipes importantes

  SOUS-QUALIFIÉ : le candidat n'a pas encore le niveau requis pour ce poste
  QUALIFIÉ      : le candidat correspond au niveau du poste
  SUR-QUALIFIÉ  : le candidat dépasse le niveau du poste

RÈGLE ANTI-SURCLASSEMENT :
  Si SUR-QUALIFIÉ → score plafonné à 6/10 pour CE poste.
  Ne jamais donner 8+/10 à un candidat sur-qualifié pour un poste inférieur à son niveau.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 3 — SCORING STRICT (basé sur l'étape 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Score de base selon les années d'expérience DANS LE DOMAINE du poste :
  0 an                          → base 2/10
  1-2 ans                       → base 3/10
  3-5 ans                       → base 5/10
  6-10 ans dans le domaine      → base 7/10
  10+ ans dans le domaine exact → base 8/10

Modificateurs (+/-) :
  +1 : certifications/qualifications directement liées au poste "{poste}"
  +1 : expérience chez GTP ou Sonatrach spécifiquement
  +1 : formation exactement alignée avec le poste
  -1 : formation hors domaine technique pour un poste technique
  -1 : aucune compétence clé du poste "{poste}" présente dans le CV
  -2 : candidat SUR-QUALIFIÉ (score plafonné à 6 après calcul)

Plafonds absolus :
  Hors domaine            → 0/10 (fixe)
  Aucune expérience       → max 5/10
  Sur-qualifié            → max 6/10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 4 — POSTE RECOMMANDÉ SELON LE PROFIL RÉEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Détermine le poste qui correspond AU PROFIL RÉEL du candidat,
indépendamment du poste visé "{poste}".

RÈGLES STRICTES :
  1. Le poste recommandé DOIT être dans la liste GTP fournie.
  2. Le DOMAINE du poste recommandé DOIT correspondre au DOMAINE RÉEL du CV
     (un comptable → poste finance/gestion ; un soudeur → poste soudage/tuyauterie)
  3. Niveau selon l'expérience :
       0-2 ans  → poste DÉBUTANT/JUNIOR dans le domaine du CV
       3-7 ans  → poste CONFIRMÉ dans le domaine du CV
       8-14 ans → poste SENIOR ou CHEF D'ÉQUIPE (seulement si gestion documentée)
       15+ ans  → poste CHEF DE DÉPARTEMENT possible si gestion prouvée
  4. Chef d'équipe ou poste de management → UNIQUEMENT si le CV mentionne
     explicitement une expérience d'encadrement/supervision d'équipe.
  5. Si profil totalement hors domaine GTP → "Profil non compatible GTP"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 5 — ADÉQUATION DU POSTE VISÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Si un poste "{poste}" a été fourni, évalue s'il est adapté au profil réel :

  ADAPTÉ      : poste visé correspond au niveau ET au domaine réel du candidat
  TROP ÉLEVÉ  : le candidat vise un poste supérieur à son niveau actuel
  TROP BAS    : le candidat est sur-qualifié pour le poste visé
  HORS DOMAINE: le poste visé n'a aucun rapport avec le domaine réel du candidat

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE RÉPONSE OBLIGATOIRE (en français)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**INVENTAIRE DU CV**
- Domaine réel du CV : [domaine exact]
- Expérience totale : [X années] dans [domaine(s)]
- Formation : [niveau + domaine]
- Gestion d'équipe : [OUI — X personnes / NON]
- Certifications clés : [liste ou "Aucune"]
- Niveau candidat vs poste "{poste}" : [SOUS-QUALIFIÉ / QUALIFIÉ / SUR-QUALIFIÉ / HORS DOMAINE]
- Année dernier diplôme : [AAAA]

**SCORE DE CORRESPONDANCE** : [0-10]/10
(Score de base : X/10 | Modificateurs : [détail] | Résultat final : Y/10)

**ADÉQUATION DU POSTE VISÉ**
[ADAPTÉ / TROP ÉLEVÉ / TROP BAS / HORS DOMAINE] — [explication 1-2 phrases]

**POINTS FORTS**
- [atouts réels du candidat pour le poste "{poste}"]

**POINTS FAIBLES / MANQUANTS**
- [lacunes par rapport aux exigences de "{poste}"]

**RECOMMANDATION FINALE**
[Recommandé / À étudier / Non recommandé] — [justification courte]

**REMARQUES**
[Observations utiles, notamment si le candidat mérite un poste différent]

**POSTE RECOMMANDÉ** : [titre EXACT de la liste GTP correspondant au domaine ET au niveau RÉEL du candidat]
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
