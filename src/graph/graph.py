"""
build_rag_graph() — construit et compile le StateGraph LangGraph.

Appelé une fois dans RAGPipeline.__init__() après l'initialisation
de tous les composants. Stocké dans self._graph et invoqué dans query().
"""
import logging
from functools import partial
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import build_nodes, route_node

logger = logging.getLogger(__name__)


def build_rag_graph(components: Dict[str, Any]):
    """
    Construit et compile le graphe LangGraph.

    Parameters
    ----------
    components : dict
        Instances des composants du pipeline. Clés requises :
        query_transformer, intent_router, embedder, vector_store,
        bm25, reranker, llm, structured, schema, config

    Returns
    -------
    CompiledGraph
        Graphe LangGraph compilé qui accepte un GraphState en entrée
        et retourne un GraphState final.
    """
    nodes      = build_nodes(components)
    schema     = components["schema"]
    structured = components["structured"]

    _route = partial(route_node, schema=schema, structured=structured)

    builder = StateGraph(GraphState)

    # Enregistrement des nœuds
<<<<<<< HEAD
    builder.add_node("contextualize",        nodes["contextualize_node"])
    builder.add_node("intent",               nodes["intent_node"])
    builder.add_node("structured_qa_direct", nodes["structured_qa_direct_node"])
    builder.add_node("exhaustive",           nodes["exhaustive_node"])
    builder.add_node("structured_qa",        nodes["structured_qa_node"])
    builder.add_node("retrieve",             nodes["retrieve_node"])
    builder.add_node("rerank",               nodes["rerank_node"])
    builder.add_node("generate",             nodes["generate_node"])
    builder.add_node("finalize",             nodes["finalize_node"])
=======
    builder.add_node("contextualize", nodes["contextualize_node"])
    builder.add_node("intent",        nodes["intent_node"])
    builder.add_node("exhaustive",    nodes["exhaustive_node"])
    builder.add_node("structured_qa", nodes["structured_qa_node"])
    builder.add_node("retrieve",      nodes["retrieve_node"])
    builder.add_node("rerank",        nodes["rerank_node"])
    builder.add_node("generate",      nodes["generate_node"])
    builder.add_node("finalize",      nodes["finalize_node"])
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000

    # Arêtes linéaires
    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "intent")

    # Routage conditionnel après classification d'intent
    builder.add_conditional_edges(
        "intent",
        _route,
        {
<<<<<<< HEAD
            "exhaustive_path":           "exhaustive",
            "structured_qa_path":        "structured_qa",
            "structured_qa_direct_path": "structured_qa_direct",
            "rag_path":                  "retrieve",
=======
            "exhaustive_path":    "exhaustive",
            "structured_qa_path": "structured_qa",
            "rag_path":           "retrieve",
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
        },
    )

    # Chemin A : exhaustive → finalize → END
    builder.add_edge("exhaustive", "finalize")

    # Chemin B : structured_qa → finalize → END
    builder.add_edge("structured_qa", "finalize")

<<<<<<< HEAD
    # Chemin D : structured_qa_direct → finalize → END
    builder.add_edge("structured_qa_direct", "finalize")

=======
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
    # Chemin C : retrieve → rerank → generate → finalize → END
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank",   "generate")
    builder.add_edge("generate", "finalize")

    # Tous les chemins convergent vers finalize → END
    builder.add_edge("finalize", END)

    compiled = builder.compile()
    logger.info("LangGraph: graphe RAG compilé avec succès.")
<<<<<<< HEAD
    return compiled
=======
    return compiled
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
