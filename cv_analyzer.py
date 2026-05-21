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
# Prompt analyse — mode STANDARD (poste précisé)
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
RÈGLES RH STRICTES
════════════════════════════════════════

1. Évaluer UNIQUEMENT les informations explicitement présentes dans le CV.
2. Ne jamais inventer de compétences.
3. Ne jamais supposer une expérience.
4. Les langues, soft skills et qualités générales ne compensent PAS
   l'absence de compétences techniques du poste.
5. Si le domaine principal du CV est incompatible avec le poste :
   → SCORE MAXIMUM = 2/10
6. Si le CV est totalement hors domaine :
   → SCORE = 0/10
   → DÉCISION = Non recommandé
7. Un diplôme seul sans expérience pertinente :
   → score maximum = 6/10
8. Une expérience réelle directement liée au poste est obligatoire
   pour obtenir un score élevé.
9. Les stages comptent faiblement comme expérience.
10. Expérience Oil & Gas / Sonatrach :
   → bonus +1

════════════════════════════════════════
DÉTECTION DU DOMAINE (TRÈS IMPORTANT)
════════════════════════════════════════

Identifier d'abord le domaine principal du candidat.

Exemples de domaines :
- Informatique / Développement
- Réseaux / Systèmes
- Cybersécurité
- Data / IA
- Comptabilité / Finance
- Juridique
- HSE
- Soudage
- Maintenance industrielle
- Génie civil
- RH / Administration

Comparer ensuite ce domaine avec le poste demandé.

════════════════════════════════════════
INCOMPATIBILITÉS ABSOLUES
════════════════════════════════════════

Ces cas doivent être NOTÉS ENTRE 0 ET 2 MAXIMUM :

- Soudeur ↔ Développeur informatique
- Comptable ↔ Développeur
- Juriste ↔ Réseaux
- RH ↔ Cybersécurité
- HSE ↔ Développeur
- Génie civil ↔ Data Scientist
- Maintenance mécanique ↔ Développeur logiciel

Dans ces cas :
- DOMAINE = Incompatible
- DÉCISION = Non recommandé
- SCORE = 0 à 2 maximum

════════════════════════════════════════
BARÈME OBLIGATOIRE
════════════════════════════════════════

1. Compatibilité domaine (0 à 3 points)

3 = domaine parfaitement adapté
2 = domaine proche
1 = domaine partiellement lié
0 = domaine incompatible

----------------------------------------

2. Diplôme / Formation (0 à 2 points)

2 = diplôme directement lié au poste
1 = formation partiellement liée
0 = aucun diplôme pertinent

----------------------------------------

3. Compétences techniques (0 à 2 points)

Comparer les compétences du CV
avec les exigences du poste.

2 = compétences solides et pertinentes
1 = compétences partielles
0 = compétences absentes

----------------------------------------

4. Expérience professionnelle (0 à 2 points)

2 = expérience forte et pertinente
1 = expérience limitée
0 = aucune expérience pertinente

----------------------------------------

5. Bonus secteur Oil & Gas (0 à 1 point)

1 = expérience pétrole/gaz/Sonatrach
0 = aucune

════════════════════════════════════════
INTERPRÉTATION DU SCORE
════════════════════════════════════════

0-2 :
Profil incompatible ou hors domaine

3-4 :
Faible adéquation

5-6 :
Adéquation moyenne

7-8 :
Bonne adéquation

9-10 :
Excellente adéquation

════════════════════════════════════════
RÈGLES POUR LE POSTE RECOMMANDÉ
════════════════════════════════════════

Le POSTE RECOMMANDÉ doit être basé UNIQUEMENT
sur le domaine réel du CV.

NE JAMAIS recopier automatiquement le poste demandé.

Si le candidat est incompatible avec le poste cible :
- proposer 2 à 3 postes cohérents avec son vrai domaine
- chaque poste sur une ligne séparée commençant par "- "

Exemples obligatoires :

CV comptabilité
→ - Comptable junior
→ - Auditeur interne
→ - Contrôleur de gestion

CV soudage
→ - Soudeur industriel
→ - Technicien chaudronnerie
→ - Chef d'équipe soudage

CV lettres / langues
→ - Assistant administratif
→ - Chargé de communication
→ - Traducteur technique

CV HSE
→ - Ingénieur HSE
→ - Agent sécurité industrielle

CV juridique
→ - Juriste d'entreprise
→ - Chargé de conformité

Si le candidat est compatible avec le poste cible :
- proposer le poste demandé en premier
- ajouter 1 à 2 variantes proches

════════════════════════════════════════
FORMAT DE RÉPONSE OBLIGATOIRE
════════════════════════════════════════

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

**POSTE RECOMMANDÉ** :
- poste 1
- poste 2
- poste 3 (si pertinent)

**ANNÉES_EXPÉRIENCE** : entier ou -1
**ANNÉE_DIPLOME** : année ou 0

════════════════════════════════════════
IMPORTANT
════════════════════════════════════════

Ne jamais être "gentil".
Être STRICT comme un vrai recruteur Sonatrach.

Un candidat hors domaine ne doit JAMAIS recevoir
un score moyen ou élevé.

NE JAMAIS afficher le barème de notation dans la réponse.
NE JAMAIS expliquer les points attribués.
Répondre UNIQUEMENT avec le format demandé ci-dessus.
"""


# ─────────────────────────────────────────────────────────────
# Prompt analyse — mode DÉCOUVERTE (aucun poste fourni)
# ─────────────────────────────────────────────────────────────

DISCOVERY_PROMPT = """
Tu es un expert RH senior chez Sonatrach spécialisé dans le recrutement technique.

Un candidat soumet son CV sans préciser de poste visé.
Ta mission : analyser son profil réel et identifier les postes Sonatrach les mieux adaptés.

════════════════════════════════════════
CV DU CANDIDAT
════════════════════════════════════════

{cv_text}

════════════════════════════════════════
RÉFÉRENTIEL POSTES SONATRACH
════════════════════════════════════════

{job_context}

════════════════════════════════════════
RÈGLES STRICTES
════════════════════════════════════════

1. Identifier le DOMAINE PRINCIPAL du candidat (formation + expérience réelle).
2. Évaluer UNIQUEMENT ce qui est explicitement présent dans le CV.
3. Ne jamais inventer de compétences ou supposer une expérience.
4. Les postes recommandés DOIVENT correspondre au vrai domaine du CV.
5. Proposer 2 à 3 postes Sonatrach réellement compatibles, du plus adapté au moins adapté.
6. Le score reflète l'employabilité générale du candidat chez Sonatrach
   (expérience, diplôme, pertinence du profil).

════════════════════════════════════════
EXEMPLES DE CORRESPONDANCES OBLIGATOIRES
════════════════════════════════════════

CV maintenance industrielle :
→ - Technicien de maintenance industrielle
→ - Agent de maintenance équipements
→ - Chef d'équipe maintenance

CV informatique / développement :
→ - Développeur logiciel
→ - Ingénieur systèmes d'information
→ - Analyste programmeur

CV réseaux / télécoms :
→ - Ingénieur réseaux télécoms
→ - Technicien infrastructure réseau
→ - Administrateur systèmes

CV comptabilité / finance :
→ - Comptable
→ - Contrôleur de gestion
→ - Auditeur interne

CV HSE :
→ - Ingénieur HSE
→ - Agent sécurité industrielle
→ - Technicien environnement

CV génie civil / BTP :
→ - Ingénieur génie civil
→ - Conducteur de travaux
→ - Technicien topographie

CV RH / administration :
→ - Chargé de ressources humaines
→ - Assistant RH
→ - Gestionnaire de paie

CV juridique :
→ - Juriste d'entreprise
→ - Chargé de conformité réglementaire

CV soudage / chaudronnerie :
→ - Soudeur industriel
→ - Technicien chaudronnerie
→ - Inspecteur soudage

════════════════════════════════════════
FORMAT DE RÉPONSE OBLIGATOIRE
════════════════════════════════════════

**SCORE** : X/10
**DOMAINE** : [domaine principal détecté]
**DÉCISION** : Recommandé / À étudier / Non recommandé

**ATOUTS**
- point 1
- point 2
- point 3

**LACUNES**
- point 1
- point 2

**POSTE RECOMMANDÉ** :
- Poste 1 (le plus adapté)
- Poste 2
- Poste 3 (si pertinent)

**ANNÉES_EXPÉRIENCE** : entier ou -1
**ANNÉE_DIPLOME** : année ou 0

════════════════════════════════════════
IMPORTANT
════════════════════════════════════════

NE JAMAIS afficher le barème de notation dans la réponse.
NE JAMAIS expliquer les points attribués.
NE JAMAIS proposer un poste informatique à un soudeur.
NE JAMAIS proposer un poste technique à un juriste ou comptable.
Répondre UNIQUEMENT avec le format demandé ci-dessus.
Être CONCIS et DIRECT.
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    """
    Choisit le bon prompt selon que le poste est précisé ou non.
    - Avec poste  → ANALYSIS_PROMPT  (évaluation par rapport au poste cible)
    - Sans poste  → DISCOVERY_PROMPT (découverte des postes compatibles)
    """
    fallback_context = (
        "Aucun document spécifique trouvé dans la base. "
        "Appliquer les critères RH généraux Sonatrach pour ce type de profil."
    )

    if not poste:
        return DISCOVERY_PROMPT.format(
            cv_text=cv_text[:4000],
            job_context=job_context or fallback_context,
        )

    return ANALYSIS_PROMPT.format(
        poste=poste,
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

    # ── Construction de la requête RAG ─────────────────────────
    if search_poste:
        # Mode standard : chercher les exigences du poste ciblé
        search_query = f"exigences compétences diplômes requis poste {search_poste}"
    else:
        # Mode découverte : chercher les postes compatibles avec le profil du CV
        cv_hint = " ".join(cv_text[:800].split())[:400]
        search_query = (
            f"postes Sonatrach disponibles compatibles profil candidat {cv_hint}"
        )

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
                "Sois CONCIS et DIRECT : le recruteur doit pouvoir lire la fiche en 20 secondes. "
                "Utilise UNIQUEMENT le format demandé avec des tirets. "
                "Aucune phrase de remplissage, aucun développement inutile. "
                "N'affiche JAMAIS le barème de notation ni les points attribués par critère. "
                "Base-toi UNIQUEMENT sur le contenu explicite du CV, sans invention."
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
    """
    Extrait le ou les postes recommandés depuis la réponse LLM.
    Supporte :
      - plusieurs postes listés avec "- " sur des lignes séparées
      - un poste unique sur la même ligne que le label
    Retourne une chaîne avec les postes séparés par " / ".
    """
    import re

    # ── Tentative multi-lignes ─────────────────────────────────
    # Cherche le bloc après "POSTE RECOMMANDÉ :" jusqu'au prochain label **...**
    m = re.search(
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]?\s*\n([\s\S]*?)"
        r"(?=\n\s*\*\*[A-ZÀÂÉÈÊËÎÏÔÙÛÜ_\s]+\*\*|\Z)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        block = m.group(1)
        lines = []
        for line in block.splitlines():
            cleaned = line.strip().lstrip("-•* \t")
            # Ignorer les lignes vides, trop courtes, ou qui ressemblent à des labels
            if (
                cleaned
                and len(cleaned) > 3
                and not cleaned.upper().startswith("ANNÉES")
                and not cleaned.upper().startswith("ANNÉE_")
                and not cleaned.startswith("**")
                and "[" not in cleaned
            ):
                lines.append(cleaned)
        if lines:
            return " / ".join(lines)

    # ── Fallback : poste unique sur la même ligne que le label ─
    patterns = [
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]{4,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•-[] \t")
            if value and len(value) > 3 and "[" not in value:
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
