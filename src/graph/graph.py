"""
build_rag_graph() — construit et compile le StateGraph LangGraph.
"""
import logging
from functools import partial
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import build_nodes, route_node

logger = logging.getLogger(__name__)


def build_rag_graph(components: Dict[str, Any]):
    nodes      = build_nodes(components)
    schema     = components["schema"]
    structured = components["structured"]

    _route = partial(route_node, schema=schema, structured=structured)

    builder = StateGraph(GraphState)

    # Enregistrement des nœuds
    builder.add_node("contextualize",        nodes["contextualize_node"])
    builder.add_node("intent",               nodes["intent_node"])
    builder.add_node("structured_qa_direct", nodes["structured_qa_direct_node"])
    builder.add_node("exhaustive",           nodes["exhaustive_node"])
    builder.add_node("structured_qa",        nodes["structured_qa_node"])
    builder.add_node("retrieve",             nodes["retrieve_node"])
    builder.add_node("rerank",               nodes["rerank_node"])
    builder.add_node("generate",             nodes["generate_node"])
    builder.add_node("finalize",             nodes["finalize_node"])

    # Arêtes linéaires
    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "intent")

    # Routage conditionnel après classification d'intent
    builder.add_conditional_edges(
        "intent",
        _route,
        {
            "exhaustive_path":           "exhaustive",
            "structured_qa_path":        "structured_qa",
            "structured_qa_direct_path": "structured_qa_direct",
            "rag_path":                  "retrieve",
        },
    )

    # Chemin A : exhaustive → finalize → END
    builder.add_edge("exhaustive", "finalize")

    # Chemin B : structured_qa → finalize → END
    builder.add_edge("structured_qa", "finalize")

    # Chemin D/D' : structured_qa_direct → finalize (si résultat trouvé)
    #                                     → retrieve  (si aucun résultat Excel → fallback RAG)
    def _route_after_direct(state: GraphState) -> str:
        return "retrieve" if state.get("needs_rag_fallback") else "finalize"

    builder.add_conditional_edges(
        "structured_qa_direct",
        _route_after_direct,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )

    # Chemin C (RAG) : retrieve → rerank → generate → finalize → END
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank",   "generate")
    builder.add_edge("generate", "finalize")

    # Tous les chemins convergent vers finalize → END
    builder.add_edge("finalize", END)

    compiled = builder.compile()
    logger.info("LangGraph: graphe RAG compilé avec succès.")

    return compiled