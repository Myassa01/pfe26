from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # Inputs
    question: str
    history: Optional[List[Dict]]

    # Contextualization
    resolved_question: str

    # Intent
    intent_data: Dict[str, Any]

    # Retrieval
    search_query: str
    hybrid_results: List[Dict]
    reranked_chunks: List[Dict]
    filtered_chunks: List[Dict]

    # Generation
    context: str
    answer: str

    # Output metadata
    sources: List[str]
    chunks_used: int
    elapsed_seconds: float
    warnings: List[str]
<<<<<<< HEAD
    path_taken: str  # "exhaustive" | "structured_qa" | "semantic_rag"
=======
    path_taken: str  # "exhaustive" | "structured_qa" | "semantic_rag" | "empty_index"
>>>>>>> 523536e19cd5c29d340be65ba01ccf0c173c0000
