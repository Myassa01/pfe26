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
# Prompt analyse
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH senior chez GTP (Groupe Travaux Pétroliers), spécialisé dans 
l'évaluation et le classement de profils techniques et ingénierie dans le secteur 
pétrolier et de la construction industrielle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Poste cible évalué : "{poste}"

Tu dois :
1. Analyser ce CV par rapport au poste "{poste}"
2. Donner un score de correspondance STRICT selon le barème ci-dessous
3. Identifier le poste GTP officiel qui correspond le mieux à ce profil
4. Si le poste du CV est SUPÉRIEUR au poste cible : expliquer pourquoi ce profil 
   reste pertinent mais surdimensionné
5. Si le poste du CV est INFÉRIEUR au poste cible : expliquer le manque

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXIGENCES DU POSTE (depuis référentiel GTP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{job_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CV DU CANDIDAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{cv_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BARÈME DE SCORING STRICT (0-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORRESPONDANCE EXACTE (poste = poste cible) :
  10/10 → Profil idéal, toutes exigences remplies, expérience optimale
   9/10 → Profil excellent, 1-2 lacunes mineures
   8/10 → Très bonne adéquation, lacunes surmontables rapidement
   7/10 → Bonne adéquation, formation OK, expérience légèrement insuffisante

PROFIL SUPÉRIEUR (poste du CV > poste cible, ex: Ingénieur Principal pour poste Ingénieur) :
   7/10 → Surdimensionné mais mobilisable, risque de désengagement
   6/10 → Trop surdimensionné, inadapté sauf besoin de montée en charge rapide

PROFIL INFÉRIEUR (poste du CV < poste cible, ex: Technicien pour poste Ingénieur) :
   5/10 → Légèrement en dessous, potentiel évolutif fort
   4/10 → En dessous des exigences, manques importants
   3/10 → Profil clairement insuffisant pour le poste

HORS DOMAINE (ex: comptable pour poste technique, biologiste pour soudeur) :
   0-2/10 → Profil totalement inadapté, domaine sans lien

RÈGLES ABSOLUES :
✗ La motivation ou qualités personnelles seules ne font JAMAIS monter le score
✗ Diplôme non pertinent pour le domaine = score max 3/10
✗ Zéro expérience industrie pétrolière/construction = pénalité obligatoire (-1 à -2)
✗ Un profil surdimensionné NE PEUT PAS dépasser 7/10 pour un poste inférieur
✓ Expérience GTP ou Sonatrach directe = bonus +1 (dans la limite de 10)
✓ Certifications spécifiques au poste = bonus +0.5 par certification pertinente

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE RÉPONSE OBLIGATOIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Réponds STRICTEMENT en français avec ce format exact, sans déviation :

**SCORE DE CORRESPONDANCE** : [0-10]/10

**NIVEAU DU PROFIL PAR RAPPORT AU POSTE CIBLE**
[EXACT / SUPÉRIEUR / INFÉRIEUR / HORS DOMAINE] — justification en 1 phrase

**POINTS FORTS**
- [point fort 1, lié directement aux exigences du poste]
- [point fort 2]
- [point fort 3 minimum]

**POINTS FAIBLES / MANQUANTS**
- [manque 1 par rapport aux exigences du poste]
- [manque 2]

**ADÉQUATION HIÉRARCHIQUE**
Poste actuel/équivalent du candidat : [poste GTP le plus proche du profil réel]
Poste cible demandé              : {poste}
Écart                            : [Exact / +N niveau(x) au-dessus / -N niveau(x) en dessous]

**RECOMMANDATION FINALE**
[Recommandé / À étudier / Non recommandé] — justification courte et concrète.

**REMARQUES**
[Observations RH supplémentaires : risque de sur-qualification, potentiel d'évolution,
points à vérifier en entretien, etc.]

**POSTE RECOMMANDÉ** : [EXACTEMENT un poste de la liste officielle GTP ci-dessous]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LISTE OFFICIELLE DES POSTES GTP — CHOIX OBLIGATOIRE
(Le POSTE RECOMMANDÉ doit être EXACTEMENT l'un de ces postes, orthographe identique)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{postes_liste}
"""

RANKING_PROMPT = """
Tu es un expert RH senior chez GTP (Groupe Travaux Pétroliers).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION : CLASSEMENT DE CANDIDATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Poste à pourvoir : "{poste}"

Tu as analysé {n_cvs} candidats. Voici les résultats individuels :
{analyses_individuelles}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES DE CLASSEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITÉ DE CLASSEMENT (dans l'ordre) :
1. EXACT : profil dont le niveau correspond exactement au poste → classé EN PREMIER
2. LÉGÈREMENT INFÉRIEUR (-1 niveau) : profil avec potentiel d'évolution → classé 2ème
3. LÉGÈREMENT SUPÉRIEUR (+1 niveau) : profil surdimensionné mais mobilisable → classé 3ème
4. TRÈS INFÉRIEUR (-2 niveaux ou plus) → classé après
5. TRÈS SUPÉRIEUR (+2 niveaux ou plus) → classé en dernier (sur-qualification forte)
6. HORS DOMAINE → éliminé du classement

À SCORE ÉGAL, départage par :
- Expérience directe GTP ou Sonatrach
- Nombre d'années d'expérience dans le domaine
- Certifications pertinentes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE RÉPONSE OBLIGATOIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**CLASSEMENT FINAL — POSTE : {poste}**

| Rang | Candidat | Score | Niveau profil | Justification courte |
|------|----------|-------|---------------|----------------------|
|  1   | [Nom]    | [X/10]| [EXACT/SUP/INF]| [1 phrase max]      |
|  2   | ...      | ...   | ...           | ...                  |

**CANDIDAT RECOMMANDÉ EN PRIORITÉ** : [Nom — Score — Justification 2 phrases]

**CANDIDATS À ÉTUDIER** : [Liste avec raisons]

**CANDIDATS NON RECOMMANDÉS** : [Liste avec raisons]

**NOTE RH** : [Observation globale sur le vivier de candidats pour ce poste]
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

def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str) -> dict:
    import time
    t0 = time.time()

    search_poste = poste.strip() if poste else ""

    if search_poste:
        search_query = f"exigences compétences diplômes requis poste {search_poste}"
    else:
        cv_hint = " ".join(cv_text[:600].split())[:300]
        search_query = f"poste Sonatrach requis diplôme expérience {cv_hint}"

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

    # Prompt final
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
    )

    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system="Tu es un expert RH chez Sonatrach. Réponds uniquement en français.",
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )

    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score = _extract_score(answer)
    recommended_poste = _extract_recommended_poste(answer)

    
    elapsed = round(time.time() - t0, 2)

    return {
    "answer": answer,
    "score": score,
    "poste": search_poste or recommended_poste or "Non précisé",
    "recommended_poste": recommended_poste or "Non précisé",  # ← ajouter
    "sources": sources,
    "elapsed_seconds": elapsed,
}

# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
    import re

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
    import re
    patterns = [
        # Inline avec deux-points : **POSTE RECOMMANDÉ** : Ingénieur forage
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        # Ligne suivante : **POSTE RECOMMANDÉ**\nIngénieur forage
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            # Rejeter si c'est le texte de template (crochets, trop court)
            if value and "[" not in value and len(value) > 3:
                return value
   
    return None
