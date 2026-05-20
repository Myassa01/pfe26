"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
<<<<<<< HEAD

Architecture v3:
- SYSTEM_PROMPT (règles + format) séparé du USER PROMPT (données)
  → les règles ne sont jamais tronquées par la limite 12 000 chars du LLM
- _extract_cv_facts() : faits vérifiables extraits par regex AVANT le LLM
  → ancre le LLM sur des données réelles, empêche la hallucination
- 3 requêtes RAG ciblées au lieu d'1 générique
  → meilleure récupération des exigences depuis le PDF référentiel
- Budgets de taille stricts : job_context≤2500, cv_text≤4000
- Score plafonné en post-traitement : hors domaine=0, CV incomplet≤2
=======
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
"""

import io
import logging
<<<<<<< HEAD
import re
=======
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Extraction texte
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
<<<<<<< HEAD
=======
    """Extraction texte PDF."""
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
<<<<<<< HEAD
=======

>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    except ImportError:
        logger.warning("PyMuPDF absent, fallback pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
<<<<<<< HEAD
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
        except ImportError:
            raise RuntimeError(
                "Aucune librairie PDF disponible. Installez : pip install pymupdf"
=======
                    page.extract_text() or ""
                    for page in pdf.pages
                ).strip()
        except ImportError:
            raise RuntimeError(
                "Aucune librairie PDF disponible. "
                "Installez : pip install pymupdf"
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
            )


def extract_text_from_docx(content: bytes) -> str:
<<<<<<< HEAD
=======
    """Extraction DOCX."""
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise RuntimeError(
<<<<<<< HEAD
            "python-docx non installé. Installez : pip install python-docx"
=======
            "python-docx non installé. "
            "Installez : pip install python-docx"
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
        )


def extract_cv_text(content: bytes, filename: str) -> str:
<<<<<<< HEAD
    name = filename.lower()
=======
    """Dispatcher extraction."""
    name = filename.lower()

>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()
<<<<<<< HEAD
    raise ValueError(
        f"Format non supporté : {filename}. Formats acceptés : PDF, DOCX, TXT"
=======

    raise ValueError(
        f"Format non supporté : {filename}. "
        "Formats acceptés : PDF, DOCX, TXT"
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    )


# ─────────────────────────────────────────────────────────────
<<<<<<< HEAD
# Extraction des titres de postes GTP depuis les chunks RAG
# ─────────────────────────────────────────────────────────────

def extract_gtp_post_titles(chunks: list) -> list[str]:
    """Extrait les titres de postes GTP valides depuis les chunks RAG."""
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


# ─────────────────────────────────────────────────────────────
# Détection domaine
# ─────────────────────────────────────────────────────────────

IT_KEYWORDS = {
    "informatique", "it", "informaticien", "developer", "programmeur",
    "python", "java", "c++", "sql", "database", "réseau", "network",
    "systèmes", "linux", "windows server", "cybersécurité", "sécurité",
    "cloud", "devops", "aws", "kubernetes", "docker", "api", "web",
}

PETROLEUM_KEYWORDS = {
    "pétrolier", "petrole", "oil", "gaz", "génie pétrolier", "forage",
    "drilling", "well control", "iwcf", "downhole", "reservoir",
}


def _is_domain_compatible(cv_text: str, target_role: str) -> bool:
    cv_lower = cv_text.lower()
    role_lower = target_role.lower()

    if any(kw in role_lower for kw in IT_KEYWORDS):
        it_matches = sum(1 for kw in IT_KEYWORDS if kw in cv_lower)
        if it_matches < 2:
            has_it_studies = any(
                term in cv_lower for term in [
                    "ingénierie informatique", "génie logiciel", "informatique",
                    "licence informatique", "bac informatique",
                ]
            )
            if not has_it_studies:
                return False

    if any(kw in role_lower for kw in PETROLEUM_KEYWORDS):
        petroleum_matches = sum(1 for kw in PETROLEUM_KEYWORDS if kw in cv_lower)
        if petroleum_matches == 0:
            if "génie pétrolier" not in cv_lower:
                return False

    return True


# ─────────────────────────────────────────────────────────────
# Détection CV incomplet
# ─────────────────────────────────────────────────────────────

def _check_cv_completeness(cv_text: str) -> dict:
    """
    Détecte si un CV est incomplet ou quasi-vide.
    Utilise des frontières de mots (\\b) pour éviter les faux positifs.
    Supporte FR, EN et arabe.
    """
    word_count = len(cv_text.split())
    cv_lower = cv_text.lower()
    issues = []

    if word_count < 80:
        issues.append(f"CV très court ({word_count} mots)")

    has_year = bool(re.search(r"\b(19|20)\d{2}\b", cv_text))

    exp_patterns = [
        r"\bexpérience\b", r"\bexperience\b", r"\bemploi\b", r"\btravail\b",
        r"\bmission\b", r"\bposte\b", r"\bfonction\b",
        r"\bwork\b", r"\bjob\b", r"\bposition\b", r"\bemployment\b",
        r"\bpresent\b", r"\bactuel\b",
    ]
    has_exp_keyword = any(re.search(p, cv_lower) for p in exp_patterns)
    has_arabic_exp = bool(re.search(r"خبرة|عمل|وظيفة|مهنة", cv_text))
    has_exp = (has_year and has_exp_keyword) or has_arabic_exp

    edu_patterns = [
        r"\bformation\b", r"\bdiplôme\b", r"\bdiplome\b",
        r"\blicence\b", r"\bmaster\b", r"\bingénieur\b", r"\bdoctorat\b",
        r"\bbaccalauréat\b", r"\bbts\b", r"\bcfpa\b",
        r"\buniversité\b", r"\bécole\b",
        r"\beducation\b", r"\bdegree\b", r"\bbachelor\b",
        r"\buniversity\b", r"\bcollege\b", r"\bgraduate\b",
    ]
    has_arabic_edu = bool(re.search(r"تكوين|تعليم|شهادة|دبلوم|جامعة|كلية", cv_text))
    has_formation = (
        any(re.search(p, cv_lower) for p in edu_patterns) or has_arabic_edu
    )

    if not has_exp and word_count < 250:
        issues.append("aucune expérience professionnelle documentée")
    if not has_formation:
        issues.append("aucune formation détectée")
    if cv_text.rstrip().endswith(("...", "…")):
        issues.append("CV possiblement tronqué")

    complete = len(issues) == 0
    warning = ("CV INCOMPLET : " + "; ".join(issues)) if issues else ""

    return {
        "complete": complete,
        "warning": warning,
        "word_count": word_count,
        "has_experience": has_exp,
        "has_formation": has_formation,
    }


# ─────────────────────────────────────────────────────────────
# Extraction de faits vérifiables (avant LLM)
# ─────────────────────────────────────────────────────────────

def _extract_cv_facts(cv_text: str) -> dict:
    """
    Extrait des faits vérifiables du CV par regex AVANT d'envoyer au LLM.
    Ces faits sont injectés comme données ancrées pour bloquer la hallucination.
    Le LLM ne peut pas contredire des faits déjà calculés par le code.
    """
    import datetime
    cv_lower = cv_text.lower()

    # Niveau de diplôme (ordre de priorité décroissant)
    degree = "Non mentionné"
    if re.search(r"\bdoctorat\b|\bphd\b|\bth[eè]se\b", cv_lower):
        degree = "Doctorat/PhD"
    elif re.search(r"\bingénieur\b|\bengineer\b", cv_lower):
        degree = "Ingénieur"
    elif re.search(r"\bmaster\b|\bm\.sc\b|\bdea\b|\bdess\b|\bm2\b", cv_lower):
        degree = "Master"
    elif re.search(r"\blicence\b|\bbachelor\b|\bl3\b|\bbsc\b", cv_lower):
        degree = "Licence"
    elif re.search(r"\bbts\b|\bdut\b|\biut\b", cv_lower):
        degree = "BTS/DUT"
    elif re.search(r"\bcfpa\b|\bcap\b|\bbep\b", cv_lower):
        degree = "CAP/BEP/CFPA"

    # Plages d'emploi : "2017 - 2019" ou "2021 - Présent"
    current_year = datetime.date.today().year
    emp_ranges = re.findall(
        r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|présent|present|actuel|aujourd)",
        cv_lower,
    )
    total_exp = 0
    for start_str, end_str in emp_ranges:
        s = int(start_str)
        e = (
            current_year
            if any(x in end_str for x in ["présent", "present", "actuel", "aujourd"])
            else int(end_str)
        )
        if 1970 <= s <= e <= current_year:
            total_exp += e - s

    # Section compétences techniques (copie directe)
    skills_section = ""
    skills_match = re.search(
        r"(?:compétences?|skills?|technologies?|outils?|langages?)[^\n]*\n((?:[^\n]+\n?){1,15})",
        cv_lower,
    )
    if skills_match:
        skills_section = skills_match.group(1)[:400]

    # Certifications reconnues
    cert_keywords = [
        "certification", "certified", "certificate", "coursera",
        "aws", "azure", "cisco", "pmp", "itil", "pl-300", "pl300",
        "ielts", "toefl", "toeic", "deep learning", "ibm",
    ]
    certs_found = [kw for kw in cert_keywords if kw in cv_lower]

    has_gtp_exp = any(x in cv_lower for x in ["gtp", "groupe travaux pétroliers"])
    has_sonatrach_exp = "sonatrach" in cv_lower

    return {
        "degree": degree,
        "total_exp_years": total_exp,
        "employment_ranges": emp_ranges,
        "has_employment_dates": len(emp_ranges) > 0,
        "skills_section_preview": skills_section,
        "certifications_found": certs_found,
        "has_gtp_exp": has_gtp_exp,
        "has_sonatrach_exp": has_sonatrach_exp,
        "word_count": len(cv_text.split()),
    }


# ─────────────────────────────────────────────────────────────
# Correspondance titre GTP
# ─────────────────────────────────────────────────────────────

def find_closest_gtp_post(candidate: str, valid_titles: list[str]) -> Optional[str]:
    if not candidate or not valid_titles:
        return None
    candidate_tokens = set(re.findall(r"\w+", candidate.lower()))
    best_title, best_score = None, 0
    for title in valid_titles:
        title_tokens = set(re.findall(r"\w+", title.lower()))
        overlap = len(candidate_tokens & title_tokens)
        score = overlap / max(len(candidate_tokens), 1)
        if score > best_score:
            best_score = score
            best_title = title
    return best_title if best_score > 0 else None


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT — règles + format (jamais tronqué par le LLM)
# Passé comme paramètre "system" à llm.generate()
# _optimize_prompt() du LLM ne tronque que "prompt", pas "system"
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert RH senior chez GTP (Groupe Travaux Pétroliers).
Tu analyses des CVs par rapport aux fiches de poste du référentiel GTP.

REGLES ABSOLUES :
1. ANTI-HALLUCINATION : utilise UNIQUEMENT les informations ECRITES dans le CV.
   Si une information est absente -> ecrire "Non mentionne". Ne JAMAIS inventer.
2. CV INCOMPLET (peu de mots, pas d'experience documentee) -> SCORE <= 2/10.
3. SUR-QUALIFIE (depasse le niveau du poste) -> SCORE <= 6/10.
4. HORS DOMAINE (aucun lien formation/experience avec le poste) -> SCORE = 0/10.
5. POSTE RECOMMANDE = titre EXACT de la liste GTP fournie, rien d'autre.

SCORING base sur l'experience documentee dans le CV :
- 0 an d'experience  -> base 2/10  (max 5/10)
- 1-2 ans            -> base 3/10
- 3-5 ans            -> base 5/10
- 6-10 ans domaine   -> base 7/10
- 10+ ans domaine    -> base 8/10
Modificateurs : +1 certs liees | +1 exp GTP/Sonatrach | +1 formation alignee
               -1 formation non technique | -1 competences cles absentes

FORMAT DE REPONSE OBLIGATOIRE (en francais) :

**DONNEES VERIFIEES**
- Experience documentee : [X ans bases sur les dates du CV, ou "0 an - aucune date d'emploi"]
- Formation : [niveau exact mentionne dans le CV]
- Competences techniques listees dans le CV : [copier exactement, ou "Aucune"]
- Certifications : [liste ou "Aucune"]
- Niveau vs poste : [SOUS-QUALIFIE / QUALIFIE / SUR-QUALIFIE / HORS DOMAINE]

**CORRESPONDANCE CV - REFERENTIEL**
Competences du CV presentes dans les exigences du poste :
- [liste ou "Aucune correspondance trouvee"]
Competences requises par le poste absentes du CV :
- [liste ou "Toutes les competences sont presentes"]

**SCORE DE CORRESPONDANCE** : [0-10]/10
(Base : X/10 | Modificateurs : [detail] | Final : Y/10)

**ADEQUATION** : [ADAPTE / TROP ELEVE / TROP BAS / HORS DOMAINE] - [1 phrase max]

**POINTS FORTS** : [liste ou "Aucun point fort identifie"]

**LACUNES** : [liste ou "Aucune lacune majeure"]

**RECOMMANDATION** : [Recommande / A etudier / Non recommande] - [1 phrase]

**REMARQUES** : [observations utiles ou "RAS"]

**POSTE RECOMMANDE** : [titre EXACT de la liste GTP]"""


# ─────────────────────────────────────────────────────────────
# USER PROMPT — données uniquement
# Peut être partiellement tronqué si trop long,
# mais les règles dans SYSTEM_PROMPT sont toujours présentes.
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """{cv_completeness_warning}
=== FAITS EXTRAITS AUTOMATIQUEMENT DU CV (ne pas contredire) ===
{cv_facts_block}

=== EXIGENCES DU POSTE "{poste}" - REFERENTIEL GTP (PDF indexe) ===
{job_context}

=== CV COMPLET DU CANDIDAT ===
{cv_text}

=== POSTES GTP VALIDES (choisir le POSTE RECOMMANDE parmi cette liste uniquement) ===
{valid_posts_list}"""


# ─────────────────────────────────────────────────────────────
# Construction du prompt utilisateur
# ─────────────────────────────────────────────────────────────

def build_analysis_prompt(
    cv_text: str,
    poste: str,
    job_context: str,
    valid_gtp_posts: list[str],
    cv_completeness: dict | None = None,
    cv_facts: dict | None = None,
) -> str:
    """
    Construit le prompt utilisateur (données uniquement).
    Les règles et le format sont dans SYSTEM_PROMPT (jamais tronqués).

    Budgets stricts pour rester sous la limite 12 000 chars :
      job_context  : 2 500 chars max
      cv_text      : 4 000 chars max
      posts_block  :   800 chars max
    """
    # Postes GTP valides
    if valid_gtp_posts:
        posts_block = "\n".join(f"  * {p}" for p in valid_gtp_posts[:20])[:800]
    else:
        posts_block = "  (Aucun poste trouve dans le referentiel)"

    # Avertissement CV incomplet
    completeness_block = ""
    if cv_completeness and not cv_completeness.get("complete"):
        completeness_block = (
            f"[AVERTISSEMENT SYSTEME] CV INCOMPLET : {cv_completeness['warning']}\n"
            f"Mots : {cv_completeness['word_count']} | "
            f"Experience : {'Oui' if cv_completeness['has_experience'] else 'NON'} | "
            f"Formation : {'Oui' if cv_completeness['has_formation'] else 'NON'}\n"
            f"-> SCORE MAXIMUM AUTORISE PAR LE SYSTEME : 2/10\n"
        )

    # Faits extraits automatiquement par regex
    facts_block = "Extraction automatique indisponible."
    if cv_facts:
        exp_ranges_str = str(cv_facts["employment_ranges"]) if cv_facts["employment_ranges"] else "aucune plage detectee"
        exp_str = f"{cv_facts['total_exp_years']} an(s) ({exp_ranges_str})"
        certs_str = ", ".join(cv_facts["certifications_found"]) or "Aucune detectee"
        bonuses = []
        if cv_facts["has_gtp_exp"]:
            bonuses.append("experience GTP")
        if cv_facts["has_sonatrach_exp"]:
            bonuses.append("experience Sonatrach")

        facts_block = (
            f"Diplome detecte     : {cv_facts['degree']}\n"
            f"Experience calculee : {exp_str}\n"
            f"Certifications      : {certs_str}\n"
            f"Bonus entreprise    : {', '.join(bonuses) or 'Aucun'}\n"
            f"Longueur CV         : {cv_facts['word_count']} mots\n"
        )
        if cv_facts.get("skills_section_preview"):
            facts_block += f"Section competences : {cv_facts['skills_section_preview'][:300]}\n"

    # Contexte RAG depuis le PDF referentiel — limité à 2500 chars
    job_ctx = (job_context or "Aucun document trouve dans le referentiel GTP.")[:2500]
    if job_context and len(job_context) > 2500:
        job_ctx += "\n[... referentiel tronque - donnees partielles ...]"

    return ANALYSIS_PROMPT.format(
        poste=poste or "poste generaliste GTP",
        job_context=job_ctx,
        cv_text=cv_text[:4000],
        valid_posts_list=posts_block,
        cv_completeness_warning=completeness_block,
        cv_facts_block=facts_block,
=======
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
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    )


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

<<<<<<< HEAD
def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str) -> dict:
    import time
    t0 = time.time()

    search_poste = poste.strip() if poste else ""

    job_context = ""
    sources = []
    valid_gtp_posts: list[str] = []

    try:
        from src.retrieval.hybrid_search import reciprocal_rank_fusion

        # ── 3 requêtes RAG ciblées pour mieux trouver les exigences du PDF ──
        # Q1 : titre exact du poste
        # Q2 : compétences/diplômes requis pour ce type de poste
        # Q3 : postes GTP compatibles avec le profil CV
        cv_summary = " ".join(cv_text[:600].split())[:300]

        if search_poste:
            queries = [
                f"Poste de base: {search_poste}",
                f"competences requises diplome formation {search_poste}",
                f"niveau exigences {search_poste} GTP referentiel",
            ]
        else:
            queries = [
                f"poste GTP requis diplome experience competences {cv_summary}",
                f"Poste de base referentiel GTP compatible profil {cv_summary}",
            ]

        all_req_chunks = []
        seen_ids = set()

        for query in queries:
            q_emb = pipeline.embedder.embed_single(query)
            q_dense = pipeline.vector_store.search(q_emb, k=pipeline.config.top_k_dense)
            q_sparse = pipeline.bm25.search(query, k=pipeline.config.top_k_sparse) if pipeline.bm25 else []
            q_fused = reciprocal_rank_fusion(q_dense, q_sparse, k=pipeline.config.rrf_k)
            for chunk in q_fused[:8]:
                cid = chunk.get("id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_req_chunks.append(chunk)

        # Reranking sur les chunks agrégés (query principale = Q1)
        main_query = queries[0]
        if pipeline.reranker and all_req_chunks:
            pairs = [(main_query, c["content"]) for c in all_req_chunks]
            rerank_scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(all_req_chunks, rerank_scores):
                c["rerank_score"] = float(s)
            all_req_chunks = sorted(
                all_req_chunks,
=======
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
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )

<<<<<<< HEAD
        # Garder les 8 meilleurs chunks pour le job_context
        req_chunks = all_req_chunks[:8]

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in req_chunks
        )
        sources = list({c["metadata"].get("source", "?") for c in req_chunks})

        if not job_context.strip():
            logger.warning(
                "RAG : aucun chunk pertinent trouve pour le poste '%s'. "
                "Verifiez que le PDF referentiel est bien indexe.",
                search_poste,
            )

        # Postes GTP valides (pour contraindre le POSTE RECOMMANDE)
        valid_gtp_posts = extract_gtp_post_titles(all_req_chunks)
        logger.info("Postes GTP extraits : %d | Chunks RAG utilises : %d",
                    len(valid_gtp_posts), len(req_chunks))

    except Exception as e:
        logger.warning("Erreur recherche RAG : %s", e)

    # ── Vérification complétude + extraction de faits ─────────────────────
    cv_completeness = _check_cv_completeness(cv_text)
    cv_facts = _extract_cv_facts(cv_text)

    if not cv_completeness["complete"]:
        logger.info("CV incomplet : %s", cv_completeness["warning"])

    logger.info(
        "CV facts — diplome=%s | exp=%d ans | certs=%s",
        cv_facts["degree"],
        cv_facts["total_exp_years"],
        cv_facts["certifications_found"],
    )

    # ── Construction du prompt ────────────────────────────────────────────
=======
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
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
<<<<<<< HEAD
        valid_gtp_posts=valid_gtp_posts,
        cv_completeness=cv_completeness,
        cv_facts=cv_facts,
    )

    # ── Appel LLM avec SYSTEM_PROMPT (règles jamais tronquées) ────────────
    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    # ── Post-traitement : corrections de score garanties par le code ───────
    is_compatible = _is_domain_compatible(cv_text, search_poste)
    score = _extract_score(answer)

    if not is_compatible:
        score = 0
        logger.info("Score corrige a 0/10 (HORS DOMAINE)")

    if not cv_completeness["complete"] and score is not None and score > 2:
        logger.info("Score corrige %d -> 2/10 (CV INCOMPLET)", score)
        score = 2

    recommended_poste_raw = _extract_recommended_poste(answer)
    recommended_poste = _validate_recommended_poste(recommended_poste_raw, valid_gtp_posts)

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "Analyse terminee en %.2fs — score=%s — poste_rec=%s",
        elapsed, score, recommended_poste,
    )

    return {
        "answer": answer,
        "score": score,
        "poste": search_poste or recommended_poste or "Non precise",
        "recommended_poste": recommended_poste or "Non precise",
        "sources": sources,
        "elapsed_seconds": elapsed,
        "valid_gtp_posts_count": len(valid_gtp_posts),
        "cv_completeness": cv_completeness,
        "cv_facts": cv_facts,
=======
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
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    }


# ─────────────────────────────────────────────────────────────
<<<<<<< HEAD
# Parsers de la réponse LLM
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
=======
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

>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    patterns = [
        r"SCORE[^:\n]*:\s*\**\s*(\d{1,2})\s*\**\s*/\s*10",
        r"SCORE[^:\n]*:\s*\[?(\d{1,2})\]?\s*/\s*10",
        r"\*\*(\d{1,2})\*\*\s*/\s*10",
<<<<<<< HEAD
        r"(\d{1,2})\s*/\s*10",
    ]
=======
        r":\s*(\d{1,2})\s*/10",
        r"(\d{1,2})\s*/\s*10",
    ]

>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 10:
                return val
<<<<<<< HEAD
=======

    # Fallback : premier X/10 dans le texte
    m = re.search(r'\b(\d{1,2})/10\b', text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val

>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
<<<<<<< HEAD
=======
    import re
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
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


<<<<<<< HEAD
def _validate_recommended_poste(
    candidate: Optional[str],
    valid_titles: list[str],
) -> Optional[str]:
    if not candidate:
        return None
    if not valid_titles:
        return candidate

    candidate_lower = candidate.strip().lower()
    for title in valid_titles:
        if title.lower() == candidate_lower:
            return title

    closest = find_closest_gtp_post(candidate, valid_titles)
    if closest:
        logger.info("Poste LLM '%s' -> corrige en '%s'", candidate, closest)
        return closest

    logger.warning("Poste recommande '%s' absent du referentiel GTP.", candidate)
    return candidate
=======
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
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
