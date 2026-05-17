"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
Les postes recommandés sont STRICTEMENT issus du Référentiel GTP.
"""

import io
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Liste officielle des postes GTP (extraite du référentiel PDF)
# Le LLM ne peut recommander QUE des postes de cette liste.
# ─────────────────────────────────────────────────────────────

POSTES_GTP = [
    "Acheteur", "Acheteur Principal", "Administrateur Base De Données",
    "Agent Administratif", "Agent D'Entretien", "Agent D'Hygiene",
    "Agent De Reprographie", "Agent De Saisie", "Agent De Surete Interne",
    "Agent De Transit", "Aide Boulanger", "Aide Cuisinier", "Aide Magasinier",
    "Aide Patissier", "Aide Soignant", "Ambulancier", "Analyste De Stocks",
    "Animateur Sport Et Loisirs", "Animateur Surete Interne", "Architecte",
    "Architecte En Chef", "Architecte Principal", "Assistant Administratif",
    "Assistant De Direction", "Assistant De Direction Generale",
    "Assistant Directeur Controle De Gestion", "Assistant Directeur Engineering",
    "Assistant Directeur Finances & Comptabilite",
    "Assistant Directeur Informatique", "Assistant Directeur Juridique",
    "Assistant Directeur Logistique", "Assistant Directeur Maintenance Industrielle",
    "Assistant Directeur Qhse", "Assistant Directeur Ressources Humaines",
    "Assistant Secretaire", "Auditeur",
    "Cadre Administratif", "Cadre Controle De Gestion", "Cadre Financier Et Comptable",
    "Cadre Informatique", "Cadre Logistique", "Cadre Technique",
    "Caissier", "Caissier Principal", "Calorifugeur", "Cariste",
    "Charge D'Etudes Administratives", "Charge D'Etudes Controle De Gestion",
    "Charge D'Etudes Finances Et Comptabilite", "Charge D'Etudes Logistique",
    "Charge D'Etudes Techniques",
    "Charge De Mission Controle De Gestion", "Charge De Mission Engineering",
    "Charge De Mission Etudes Administratives",
    "Charge De Mission Finances & Comptabilte",
    "Charge De Mission Informatique", "Charge De Mission Juridique",
    "Charge De Mission Logistique", "Charge De Mission Maintenance Industrielle",
    "Charge De Mission Qhse", "Charge De Mission Sie",
    "Chaudronnier", "Chaudronnier Hautement Qualifie",
    "Chauffeur Accompagnateur", "Chauffeur De Direction",
    "Chef Cuisinier",
    "Chef D'Equipe Electriciens Industriels", "Chef D'Equipe Electromecaniciens",
    "Chef D'Equipe Instrumentistes", "Chef D'Equipe Mecaniciens Industriels",
    "Chef D'Equipe Monteurs", "Chef D'Equipe Soudeurs",
    "Chef De Base Vie", "Chef De Camp",
    "Chef De Chantier Entretien", "Chef De Chantier Genie Civil",
    "Chef De Departement Administration & Finances",
    "Chef De Departement Approvisionnement & Moyens Communs",
    "Chef De Departement Budget", "Chef De Departement Comptabilite & Finances",
    "Chef De Departement Construction",
    "Chef De Departement Cybersecurite Et Conformite",
    "Chef De Departement Developpement Et Data Management",
    "Chef De Departement Developpement Informatique",
    "Chef De Departement Electricite/Instrumentation",
    "Chef De Departement Finances",
    "Chef De Departement Genie Civil",
    "Chef De Departement Gestion Des Stocks",
    "Chef De Departement Gestion Ressources Humaines",
    "Chef De Departement Hygiene, Securite Et Environnement",
    "Chef De Departement Infrastructure & Systemes Informatiques",
    "Chef De Departement Instrumentation Et Automatisation",
    "Chef De Departement Logistique",
    "Chef De Departement Maintenance",
    "Chef De Departement Maintenance Industrielle",
    "Chef De Departement Operations",
    "Chef De Departement Technique",
    "Chef De Projet", "Chef De Projet Engineering",
    "Chef De Projet Informatique Développement / Finances Et",
    "Chef De Projet Informatique Développement / Logistique",
    "Chef De Projet Informatique Développement / Ressources Humaines",
    "Chef De Projet Informatique Développement / Technique",
    "Chef De Projet Infrastructure Réseau Et Convergence",
    "Chef De Projet Infrastructure Système",
    "Chef De Projet Infrastructure Sécurité Opérationnelle",
    "Chef De Section Developpement",
    "Chef De Section Electricite",
    "Chef De Section Etudes",
    "Chef De Section Formation",
    "Chef De Section Gestion Des Stocks",
    "Chef De Section Instrumentation",
    "Chef De Section Maintenance",
    "Chef De Section Maintenance Industrielle",
    "Chef De Section Mecanique",
    "Chef De Section Numerisation",
    "Chef De Section Programmation",
    "Chef De Section Soudage",
    "Chef De Section Technique",
    "Chef De Section Telecommunication",
    "Chef De Service Administration",
    "Chef De Service Electricite Et Instrumentation",
    "Chef De Service Finances Et Comptabilité",
    "Chef De Service Hse",
    "Chef De Service Informatique",
    "Chef De Service Logistique",
    "Chef De Service Maintenance Industrielle",
    "Chef De Service Mecanique",
    "Chef De Service Numerisation Et Digitalisation",
    "Chef De Service Parc Informatique",
    "Chef De Service Soudage Et Controle",
    "Chef De Service Technique",
    "Chef De Service Travaux Et Maintenance Industrielle",
    "Chirurgien Dentiste",
    "Coffreur", "Comptable", "Comptable Principal",
    "Conducteur De Travaux Canalisation",
    "Conducteur De Travaux Electricite Industrielle",
    "Conducteur De Travaux Genie Civil",
    "Conducteur De Travaux Instrumentation",
    "Conducteur De Travaux Mecanique Industrielle",
    "Conducteur De Travaux Montage",
    "Controleur De Soudure",
    "Declarant En Douanes", "Depanneur En Telecommunication",
    "Dessinateur Projeteur",
    "Directeur De Projet", "Directeur Engineering",
    "Directeur Finances Et Comptabilite",
    "Directeur Maintenance Industrielle",
    "Directeur Qhse", "Directeur Ressources Humaines",
    "Directeur Systemes D'Information",
    "Documentaliste",
    "Electricien Industriel", "Electromecanicien", "Electrotechnicien",
    "Enseignant Technique",
    "Formateur",
    "Gestionnaire Administratif", "Gestionnaire De Stocks",
    "Ingenieur", "Ingenieur Automatisation", "Ingenieur Chimie",
    "Ingenieur Construction Metallique",
    "Ingenieur Contrôle Qualite", "Ingenieur Contrôle Qualite Des Projets",
    "Ingenieur Developpement Informatique",
    "Ingenieur Electricite", "Ingenieur Electromecanique",
    "Ingenieur Electronique",
    "Ingenieur En Chef Developpement Informatique",
    "Ingenieur En Chef Etudes Automatisation",
    "Ingenieur En Chef Etudes Electriques",
    "Ingenieur En Chef Etudes Genie Civil",
    "Ingenieur En Chef Etudes Instrumentation",
    "Ingenieur En Chef Etudes Mecaniques",
    "Ingenieur En Chef Genie Civil",
    "Ingenieur En Chef Hse",
    "Ingenieur En Chef Infrastructure Informatique",
    "Ingenieur En Chef Instrumentation",
    "Ingenieur En Chef Logistique",
    "Ingenieur En Chef Maintenance Industrielle",
    "Ingenieur En Chef Mecanique",
    "Ingenieur En Chef Organisation",
    "Ingenieur En Chef Pipeline",
    "Ingenieur En Chef Planning",
    "Ingenieur En Chef Process",
    "Ingenieur En Chef Soudage",
    "Ingenieur En Chef Statistiques",
    "Ingenieur Etudes Automatisation", "Ingenieur Etudes Electriques",
    "Ingenieur Etudes Genie Civil", "Ingenieur Etudes Instrumentation",
    "Ingenieur Etudes Mecaniques",
    "Ingenieur Genie Civil", "Ingenieur Hse",
    "Ingenieur Infrastructure Informatique",
    "Ingenieur Instrumentation", "Ingenieur Logistique",
    "Ingenieur Maintenance Des Equipements",
    "Ingenieur Maintenance Industrielle",
    "Ingenieur Management Integre",
    "Ingenieur Mecanique", "Ingenieur Mecanique Industrielle",
    "Ingenieur Organisation", "Ingenieur Pipeline",
    "Ingenieur Planning", "Ingenieur Principal",
    "Ingenieur Principal Developpement Informatique",
    "Ingenieur Principal Electricite",
    "Ingenieur Principal Etudes Automatisation",
    "Ingenieur Principal Etudes Genie Civil",
    "Ingenieur Principal Etudes Instrumentation",
    "Ingenieur Principal Etudes Mecaniques",
    "Ingenieur Principal Genie Civil",
    "Ingenieur Principal Hse",
    "Ingenieur Principal Infrastructure Informatique",
    "Ingenieur Principal Instrumentation",
    "Ingenieur Principal Maintenance Industrielle",
    "Ingenieur Principal Mecanique",
    "Ingenieur Principal Planning",
    "Ingenieur Principal Process",
    "Ingenieur Principal Soudage",
    "Ingenieur Principal Statistiques",
    "Ingenieur Process", "Ingenieur Soudage",
    "Ingenieur Statistiques", "Ingenieur Topographie",
    "Ingenieur Travaux",
    "Inspecteur Contrôle Qualite Des Projets",
    "Inspecteur Hse", "Inspecteur Radioprotection",
    "Instrumentiste",
    "Juriste",
    "Magasinier", "Maçon", "Mecanicien Essence Diesel", "Mecanicien Industriel",
    "Medecin De Travail", "Medecin Generaliste",
    "Metreur", "Metreur Verificateur", "Monteur",
    "Operateur En Peinture Industrielle",
    "Radiometallographe", "Receptionniste",
    "Responsable Cellule Contrôle Qualité",
    "Responsable Cellule Informatique",
    "Responsable De Cellule Administration Du Personnel",
    "Responsable De Cellule Controle De Gestion",
    "Responsable De Cellule Controle Qualite Des Projets",
    "Responsable De Cellule Des Systemes D'Information",
    "Responsable De Cellule Documentation Et Archives",
    "Responsable De Cellule Emploi Et Competences",
    "Responsable De Cellule Hse",
    "Responsable De Cellule Informatique",
    "Responsable De Cellule Logistique",
    "Responsable De Cellule Organisation",
    "Responsable De Cellule Technique",
    "Responsable Securite Des Systemes D'Information",
    "Secretaire", "Secretaire De Direction",
    "Soudeur", "Soudeur Hautement Qualifie",
    "Sous Directeur Administration Et Finances",
    "Sous Directeur Etudes",
    "Sous Directeur Gestion Materiel Et Maintenance",
    "Sous Directeur Realisation",
    "Technicien Automatisme", "Technicien Controle De Gestion",
    "Technicien Contrôle Qualite Des Projets",
    "Technicien Electricite Industrielle",
    "Technicien Electromecanique", "Technicien Electronique",
    "Technicien En Maintenance Des Equipements",
    "Technicien Finances Et Comptabilite",
    "Technicien Hse", "Technicien Informatique",
    "Technicien Instrumentation", "Technicien Logistique",
    "Technicien Maintenance En Telecommunication",
    "Technicien Maintenance Industrielle",
    "Technicien Mecanique", "Technicien Mecanique Industrielle",
    "Technicien Superieur Automatisme",
    "Technicien Superieur Construction Metallique",
    "Technicien Superieur Controle De Gestion",
    "Technicien Superieur Contrôle Qualite Des Projets",
    "Technicien Superieur Documentation Et Archive",
    "Technicien Superieur Electricite",
    "Technicien Superieur Electromecanique",
    "Technicien Superieur Electronique",
    "Technicien Superieur Etudes Automatisation",
    "Technicien Superieur Etudes Electriques",
    "Technicien Superieur Etudes Genie Civil",
    "Technicien Superieur Etudes Instrumentation",
    "Technicien Superieur Etudes Mecaniques",
    "Technicien Superieur Finances Et Comptabilite",
    "Technicien Superieur Genie Civil",
    "Technicien Superieur Gestion Des Stocks",
    "Technicien Superieur Hse",
    "Technicien Superieur Informatique",
    "Technicien Superieur Instrumentation",
    "Technicien Superieur Logistique",
    "Technicien Superieur Maintenance Des Equipements",
    "Technicien Superieur Maintenance Industrielle",
    "Technicien Superieur Mecanique",
    "Technicien Superieur Mecanique Industrielle",
    "Topographe", "Ts Statistiques",
    "Tuyauteur", "Tuyauteur Hautement Qualifie",
    "Tuyauteur Instrumentiste",
]

# Index normalisé pour matching rapide (insensible casse/accents)
def _normalize(s: str) -> str:
    s = s.lower()
    for src, dst in [("é","e"),("è","e"),("ê","e"),("à","a"),("â","a"),
                     ("î","i"),("ô","o"),("ù","u"),("û","u"),("ç","c")]:
        s = s.replace(src, dst)
    return s

_POSTES_INDEX = {_normalize(p): p for p in POSTES_GTP}

# Formate la liste pour injection dans le prompt (compacte)
_POSTES_LISTE_STR = "\n".join(f"- {p}" for p in sorted(POSTES_GTP))


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
Tu es un expert RH chez GTP (Groupe Travaux Pétroliers).

Analyse le CV ci-dessous par rapport au poste "{poste}"
en utilisant les exigences récupérées depuis la base documentaire interne.

=== EXIGENCES DU POSTE ===
{job_context}

=== CV DU CANDIDAT ===
{cv_text}

=== BARÈME STRICT ===
- 0-2 : Profil totalement hors domaine (ex: biologiste, comptable pour poste technique)
- 3-4 : Faible adéquation, formation éloignée ou expérience insuffisante
- 5-6 : Adéquation moyenne, quelques éléments correspondent
- 7-8 : Bonne adéquation, profil correspondant avec lacunes mineures
- 9-10 : Excellente adéquation, profil idéal

RÈGLES DE NOTATION STRICTES :
- Formation sans lien avec le domaine du poste => score MAXIMUM 3
- Aucune expérience dans le secteur industriel/pétrole/construction => pénalité obligatoire
- Les qualités personnelles seules ne justifient JAMAIS un bon score
- Un biologiste, un juriste ou un comptable pour un poste technique => score 0-2
- Diplôme non pertinent pour le poste ciblé => pénalité systématique

Réponds STRICTEMENT en français avec ce format exact :

**SCORE DE CORRESPONDANCE** : [0-10]/10

**POINTS FORTS**
- ...

**POINTS FAIBLES / MANQUANTS**
- ...

**RECOMMANDATION FINALE**
Recommandé / À étudier / Non recommandé — avec justification courte.

**REMARQUES**
Observations supplémentaires si nécessaire.

**POSTE RECOMMANDÉ** : [EXACTEMENT un poste de la liste officielle GTP ci-dessous — aucune invention]

=== LISTE OFFICIELLE DES POSTES GTP (choix OBLIGATOIRE parmi ces postes uniquement) ===
{postes_liste}
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste GTP",
        job_context=job_context or (
            "Aucun document spécifique trouvé. "
            "Utiliser le barème strict et le bon sens RH."
        ),
        cv_text=cv_text[:4000],
        postes_liste=_POSTES_LISTE_STR,
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
            system=(
                "Tu es un expert RH chez GTP. Réponds uniquement en français. "
                "Le POSTE RECOMMANDÉ doit être EXACTEMENT un des postes de la liste GTP fournie. "
                "Ne jamais inventer un poste qui n'existe pas dans la liste."
            ),
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score = _extract_score(answer)
    recommended_poste = _extract_and_validate_poste(answer)

    elapsed = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "score": score,
        "poste": search_poste or recommended_poste or "Non précisé",
        "recommended_poste": recommended_poste or "Non précisé",
        "sources": sources,
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


def _extract_raw_poste(text: str) -> Optional[str]:
    """Extrait le texte brut après POSTE RECOMMANDÉ."""
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


def _find_best_match_in_gtp(raw: str) -> Optional[str]:
    """
    Trouve le meilleur match dans la liste officielle GTP.
    1. Correspondance exacte normalisée
    2. Le poste GTP dont le nom normalisé est contenu dans la réponse
    3. La réponse contient les mots-clés du poste GTP
    Retourne None si aucun match suffisamment fiable.
    """
    if not raw:
        return None

    raw_norm = _normalize(raw)

    # 1. Correspondance exacte normalisée
    if raw_norm in _POSTES_INDEX:
        return _POSTES_INDEX[raw_norm]

    # 2. Le poste GTP est entièrement contenu dans la réponse du LLM
    best_match = None
    best_len = 0
    for norm_poste, poste_orig in _POSTES_INDEX.items():
        if norm_poste in raw_norm and len(norm_poste) > best_len:
            best_match = poste_orig
            best_len = len(norm_poste)

    if best_match and best_len >= 8:  # évite les matches trop courts ("agent", "chef"…)
        return best_match

    # 3. Score de mots-clés : nombre de mots du poste GTP présents dans la réponse
    raw_words = set(raw_norm.split())
    best_score = 0
    best_match = None
    for norm_poste, poste_orig in _POSTES_INDEX.items():
        poste_words = set(norm_poste.split())
        if len(poste_words) < 2:
            continue  # trop court pour être discriminant
        common = poste_words & raw_words
        score = len(common) / len(poste_words)
        if score > best_score and score >= 0.75:
            best_score = score
            best_match = poste_orig

    return best_match  # peut être None si aucun match fiable


def _extract_and_validate_poste(text: str) -> Optional[str]:
    """
    Extrait le poste recommandé ET le valide contre la liste officielle GTP.
    Si le LLM a inventé un poste, on tente de le mapper vers le plus proche.
    Si aucun match : retourne None (affichage "Non précisé" côté frontend).
    """
    raw = _extract_raw_poste(text)
    if not raw:
        return None

    validated = _find_best_match_in_gtp(raw)

    if validated:
        logger.info("Poste recommandé validé : '%s' → '%s'", raw, validated)
    else:
        logger.warning(
            "Poste recommandé '%s' absent du référentiel GTP — ignoré", raw
        )

    return validated
