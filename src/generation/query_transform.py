"""Transformation de requêtes: réécriture et expansion multi-query."""
import re
from typing import List, Dict, Optional
from .llm import HFClient

_REWRITE_PROMPT = """Reformule cette question de recherche en français, de façon claire et précise.
Réponds avec UNIQUEMENT la question reformulée, rien d'autre.

Question: {query}
Reformulation:"""

_EXPANSION_PROMPT = """Génère 3 reformulations alternatives de cette question en français.
Réponds avec UNIQUEMENT les 3 questions, une par ligne, sans numérotation.

Question: {query}
Reformulations:"""

_CONTEXTUALIZE_SYSTEM = (
    "Tu es un assistant qui reformule des questions en français. "
    "Tu retournes UNIQUEMENT la question autonome reformulée, sans explication."
)

_CONTEXTUALIZE_PROMPT = """\
Voici un historique de conversation :
{history}

Nouvelle question : {question}

Si la question contient des références implicites ("ces", "leur", "il", "eux", "les mêmes", \
"ce département", "ces personnes", etc.) qui renvoient à des éléments mentionnés dans \
l'historique, réécris la question de façon AUTONOME et COMPLÈTE sans références implicites.
Si la question est déjà autonome et claire, retourne-la telle quelle.

Règle : retourne UNIQUEMENT la question reformulée sur une seule ligne, sans guillemets.

Question autonome :"""


# Indicateurs d'une question qui dépend du contexte précédent
_CONTEXT_TRIGGERS = re.compile(
    r"\b(ces|ceux|celles|leur|leurs|il|elle|ils|elles|eux|"
    r"les m[eê]mes|ce département|cette personne|ces personnes|"
    r"ces deux|ces 2|celui-ci|celle-ci|ce dernier|cette derni[eè]re|"
    r"parmi eux|parmi elles|de ces|d'eux)\b",
    re.IGNORECASE,
)


class QueryTransformer:
    def __init__(self, llm: HFClient):
        self.llm = llm

    def contextualize(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Reformule la question pour la rendre autonome si elle contient des références à l'historique."""
        if not history or not _CONTEXT_TRIGGERS.search(question):
            return question

        history_lines = []
        for msg in history[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

        prompt = _CONTEXTUALIZE_PROMPT.format(
            history=history_text,
            question=question.strip(),
        )
        try:
            result = self.llm.generate(
                prompt=prompt,
                system=_CONTEXTUALIZE_SYSTEM,
                temperature=0.0,
                max_tokens=120,
            )
            reformulated = result.strip()
            lines = [l.strip() for l in reformulated.split("\n") if l.strip()]
            reformulated = lines[0] if lines else question
            # Rejette si trop différente (hallucination) ou trop courte
            if len(reformulated) < 5 or len(reformulated) > 4 * len(question):
                return question
            return reformulated
        except Exception:
            return question

    def rewrite(self, query: str) -> str:
        """Réécrit la requête pour améliorer la récupération."""
        try:
            result = self.llm.generate(
                prompt=_REWRITE_PROMPT.format(query=query),
                temperature=0.1,
                max_tokens=100,
            )
            # Nettoie les artefacts courants des petits modèles
            rewritten = result.strip()
            # Supprime les préfixes parasites type "**Question reformulée:**"
            rewritten = re.sub(r"^\*{0,2}[^:]*:\*{0,2}\s*", "", rewritten).strip()
            # Prend uniquement la première ligne non vide
            lines = [l.strip() for l in rewritten.split("\n") if l.strip()]
            rewritten = lines[0] if lines else query
            # Si le résultat est trop long ou bizarre, retourne l'original
            if len(rewritten) > 3 * len(query) or len(rewritten) < 5:
                return query
            return rewritten
        except Exception:
            return query

    def expand(self, query: str) -> List[str]:
        """Génère 3 formulations alternatives de la requête."""
        try:
            result = self.llm.generate(
                prompt=_EXPANSION_PROMPT.format(query=query),
                temperature=0.4,
                max_tokens=300,
            )
            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
            # Nettoie les numérotations et préfixes
            cleaned = []
            for line in lines:
                line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                if line:
                    cleaned.append(line)
            return cleaned[:3] if cleaned else [query]
        except Exception:
            return [query]