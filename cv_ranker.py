"""
Module de classement multi-candidats.
Analyse et classe plusieurs CVs pour un poste donné.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def rank_candidates(
    pipeline,
    cv_list: list[dict],  # [{"filename": "cv1.pdf", "content": b"...", "cv_text": "..."}, ...]
    target_role: str,
) -> dict:
    """
    Analyse et classe plusieurs CV pour un poste donné.
    
    Args:
        pipeline: Pipeline RAG configuré
        cv_list: Liste de CVs avec filename, content (bytes), et optionnellement cv_text
        target_role: Poste cible pour lequel classer les candidats
    
    Returns:
        Dictionnaire avec :
        - candidates: liste des candidats analysés + classés
        - best_candidate: meilleur candidat (score max)
        - ranking: liste ordonnée [#1, #2, #3...]
        - summary: résumé du classement
    """
    from cv_analyzer import analyze_cv_with_pipeline, extract_cv_text
    
    logger.info("Classement de %d CV pour le poste : %s", len(cv_list), target_role)
    
    candidates = []
    
    # Analyse chaque CV
    for idx, cv_data in enumerate(cv_list, 1):
        filename = cv_data.get("filename", f"CV_{idx}")
        
        try:
            # Extraction texte si nécessaire
            if "cv_text" not in cv_data:
                content = cv_data.get("content")
                if isinstance(content, bytes):
                    cv_text = extract_cv_text(content, filename)
                else:
                    cv_text = str(content)
            else:
                cv_text = cv_data["cv_text"]
            
            # Analyse
            result = analyze_cv_with_pipeline(pipeline, cv_text, target_role)
            
            # Enrichissement
            result["filename"] = filename
            result["cv_text_preview"] = cv_text[:500]  # Garder un extrait
            result["rank"] = None  # Sera mis à jour
            
            candidates.append(result)
            logger.info(f"  ✓ {filename} : {result.get('score', '?')}/10")
            
        except Exception as e:
            logger.error(f"  ✗ {filename} : Erreur d'analyse — {e}")
            candidates.append({
                "filename": filename,
                "score": 0,
                "answer": f"Erreur lors de l'analyse : {str(e)}",
                "error": True,
                "rank": None,
            })
    
    # Classement par score décroissant (puis par ordre d'apparition)
    ranked = sorted(
        candidates,
        key=lambda x: (-(x.get("score") or 0), candidates.index(x))
    )
    
    # Attribution des rangs
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank
    
    # Synthèse
    best = ranked[0] if ranked else None
    summary = {
        "total_candidates": len(candidates),
        "best_candidate_filename": best["filename"] if best else None,
        "best_candidate_score": best["score"] if best else None,
        "target_role": target_role,
        "top_3": [
            {
                "rank": c["rank"],
                "filename": c["filename"],
                "score": c.get("score"),
                "poste_recommande": c.get("recommended_poste"),
                "adéquation": _adequation_label(c),
            }
            for c in ranked[:3]
        ],
    }
    
    logger.info("Classement terminé : #1 = %s (%d/10)", 
                best["filename"] if best else "N/A",
                best.get("score", 0) if best else 0)
    
    return {
        "candidates": ranked,
        "best_candidate": best,
        "ranking": ranked,
        "summary": summary,
    }


def _adequation_label(c: dict) -> str:
    """Génère un label d'adéquation basé sur le score et la complétude du CV."""
    score = c.get("score")
    completeness = c.get("cv_completeness", {})

    if not completeness.get("complete", True):
        return "⚠️ CV incomplet"
    if score == 0:
        return "✗ Hors domaine"
    if score is None:
        return "? Indéterminé"
    return "✓ Domaine OK"


def get_ranking_table(ranking_result: dict) -> str:
    """Génère un tableau de classement au format Markdown."""
    candidates = ranking_result.get("candidates", [])

    if not candidates:
        return "Aucun candidat classé."

    lines = [
        "| Rang | Candidat | Score | Poste Recommandé | Adéquation |",
        "|------|----------|-------|------------------|-----------|",
    ]

    for c in candidates:
        rank = c.get("rank", "?")
        filename = c.get("filename", "?")
        score = c.get("score", "?")
        poste = c.get("recommended_poste", "Non précisé")
        adequation = _adequation_label(c)

        # Ajouter note sur-qualifié si score plafonné
        score_display = f"{score}/10"
        if score and score <= 6:
            answer = c.get("answer", "")
            if "SUR-QUALIFIÉ" in answer.upper() or "SUR-QUALIFIE" in answer.upper():
                score_display = f"{score}/10 (sur-qualifié)"

        lines.append(f"| {rank} | {filename} | {score_display} | {poste} | {adequation} |")

    # Avertissements CV incomplets sous le tableau
    incomplete = [
        c for c in candidates
        if not c.get("cv_completeness", {}).get("complete", True)
    ]
    if incomplete:
        lines.append("")
        lines.append("**⚠️ CVs incomplets détectés :**")
        for c in incomplete:
            warning = c.get("cv_completeness", {}).get("warning", "CV incomplet")
            lines.append(f"- `{c['filename']}` : {warning}")

    return "\n".join(lines)
