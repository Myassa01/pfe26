# Plan : Migration RAG Pipeline → LangGraph

## Context
Le pipeline RAG actuel est une grande classe monolithique (`RAGPipeline` dans `src/pipeline.py`). La méthode `query()` contient toute la logique de routage, récupération et génération dans un seul bloc séquentiel impératif. L'objectif est de le transformer en un graphe d'états LangGraph — chaque étape devient un nœud indépendant, le routage conditionnel devient une arête conditionnelle explicite, et l'état partagé est un `TypedDict` typé.

**Ce qui ne change PAS** : `ingest()`, `ingest_documents()`, `api.py`, le frontend, `auth.py`, tous les composants existants (LLM, embedder, vector store, BM25, reranker, DuckDB).

---

## Fichiers à créer / modifier

| Action | Fichier |
|--------|---------|
| CRÉER | `pfe26/src/graph/__init__.py` |
| CRÉER | `pfe26/src/graph/state.py` |
| CRÉER | `pfe26/src/graph/nodes.py` |
| CRÉER | `pfe26/src/graph/graph.py` |
| MODIFIER | `pfe26/src/pipeline.py` |
| MODIFIER | `pfe26/requirements.txt` |

---

## Structure du graphe

```
START
  └─► contextualize_node   (résout "ces", "eux", "leur" via LLM + historique)
        └─► intent_node     (IntentRouter.classify → intent, source, exhaustive, filter)
              └─► [conditional edge: route_fn]
                    ├─► exhaustive_node    → finalize → END
                    ├─► structured_qa_node → finalize → END
                    └─► retrieve_node → rerank_node → generate_node → finalize → END
```

---

## Étape 1 — `pfe26/requirements.txt`

Ajouter une ligne :
```
langgraph>=0.2.0,<0.3.0
```

---

## Étape 2 — `pfe26/src/graph/__init__.py`

Fichier vide (marque le package).

---

## Étape 3 — `pfe26/src/graph/state.py`

```python
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
    dense_results: List[Dict]
    sparse_results: List[Dict]
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
    path_taken: str   # "exhaustive" | "structured_qa" | "semantic_rag"
```

`total=False` → aucune clé requise à la construction, correspond au modèle partial-update de LangGraph.

---

## Étape 4 — `pfe26/src/graph/nodes.py`

Contient :
1. **Prompts** (déplacés depuis `pipeline.py`) : `_SYSTEM_PROMPT`, `_GENERATION_PROMPT`, `_GENERATION_PROMPT_LIST`
2. **Helpers module-level** (déplacés depuis les méthodes d'instance) :
   - `_format_history(history) -> str` — derniers 6 messages
   - `_format_context(chunks) -> str` — join avec séparateur
   - `_extract_sources(chunks) -> List[str]`
   - `_fold(text) -> str` — normalisation accents
   - `_normalize_stem(fname) -> str`
   - `_filter_by_source(chunks, source) -> List[Dict]`
   - `_extract_primary_value(item, source, structured_engine) -> str`
3. **`build_nodes(components: dict) -> dict`** — factory qui capture tous les composants par closure et retourne un dict de fonctions nœud
4. **`route_node(state, schema, structured) -> str`** — edge conditionnelle (utilisée avec `partial()`)

### Nœuds (signatures exactes)

```python
def contextualize_node(state: GraphState) -> dict:
    # query_transformer.contextualize(question, history)
    # → {"resolved_question": str}

def intent_node(state: GraphState) -> dict:
    # intent_router.classify(resolved_question)
    # → {"intent_data": dict}

def exhaustive_node(state: GraphState) -> dict:
    # structured.list_values(source, column, filters, distinct=True)
    # dédup + format liste numérotée
    # → {"answer", "sources", "chunks_used", "warnings", "path_taken": "exhaustive"}

def structured_qa_node(state: GraphState) -> dict:
    # structured.list_values() → LLM generate()
    # → {"answer", "sources", "chunks_used", "warnings", "path_taken": "structured_qa"}

def retrieve_node(state: GraphState) -> dict:
    # embedder.embed_single() + vector_store.search() + bm25.search() + RRF
    # → {"search_query", "dense_results", "sparse_results", "hybrid_results"}

def rerank_node(state: GraphState) -> dict:
    # reranker.rerank() + _filter_by_source()
    # → {"reranked_chunks", "filtered_chunks"}

def generate_node(state: GraphState) -> dict:
    # llm.generate() avec context + history formatés
    # → {"context", "answer", "path_taken": "semantic_rag"}

def finalize_node(state: GraphState) -> dict:
    # Si semantic_rag : extrait sources/chunks depuis filtered_chunks
    # Sinon : reprend sources/chunks_used déjà définis
    # → {"sources", "chunks_used"}
```

### Edge conditionnelle

```python
def route_node(state: GraphState, schema: dict, structured) -> str:
    # Reproduit exactement la logique de routage de query() actuelle :
    # exhaustive + source Excel + has_table → "exhaustive_path"
    # qa + source Excel + filter + has_table → "structured_qa_path"
    # sinon → "rag_path"
```

---

## Étape 5 — `pfe26/src/graph/graph.py`

```python
from langgraph.graph import StateGraph, START, END
from functools import partial
from .state import GraphState
from .nodes import build_nodes, route_node

def build_rag_graph(components: dict):
    nodes = build_nodes(components)
    _route = partial(route_node, schema=components["schema"], structured=components["structured"])

    builder = StateGraph(GraphState)

    builder.add_node("contextualize", nodes["contextualize_node"])
    builder.add_node("intent",        nodes["intent_node"])
    builder.add_node("exhaustive",    nodes["exhaustive_node"])
    builder.add_node("structured_qa", nodes["structured_qa_node"])
    builder.add_node("retrieve",      nodes["retrieve_node"])
    builder.add_node("rerank",        nodes["rerank_node"])
    builder.add_node("generate",      nodes["generate_node"])
    builder.add_node("finalize",      nodes["finalize_node"])

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "intent")
    builder.add_conditional_edges("intent", _route, {
        "exhaustive_path":    "exhaustive",
        "structured_qa_path": "structured_qa",
        "rag_path":           "retrieve",
    })
    builder.add_edge("exhaustive",    "finalize")
    builder.add_edge("structured_qa", "finalize")
    builder.add_edge("retrieve",  "rerank")
    builder.add_edge("rerank",    "generate")
    builder.add_edge("generate",  "finalize")
    builder.add_edge("finalize",  END)

    return builder.compile()
```

---

## Étape 6 — `pfe26/src/pipeline.py` (modifications ciblées)

### 6a — Ajouter import (après les imports existants)
```python
from .graph.graph import build_rag_graph
```

### 6b — Fin de `__init__()` — ajouter construction du graphe
Après `self.intent_router = IntentRouter(llm=self.llm, schema=self.schema)` :
```python
        self._graph = self._build_graph()
```
Et ajouter la méthode helper privée :
```python
    def _build_graph(self):
        return build_rag_graph({
            "query_transformer": self.query_transformer,
            "intent_router":     self.intent_router,
            "embedder":          self.embedder,
            "vector_store":      self.vector_store,
            "bm25":              self.bm25,
            "reranker":          self.reranker,
            "llm":               self.llm,
            "structured":        self.structured,
            "schema":            self.schema,
            "config":            self.config,
        })
```

### 6c — `ingest()` et `ingest_documents()` — reconstruire le graphe après reload
Après `self.intent_router = IntentRouter(llm=self.llm, schema=self.schema)` dans les deux méthodes :
```python
        self._graph = self._build_graph()
```
**Pourquoi :** les closures des nœuds capturent `schema` et `intent_router` par référence au moment de la construction. Si on réassigne `self.schema` et `self.intent_router`, il faut reconstruire le graphe pour que les nouveaux objets soient capturés.

### 6d — Remplacer le corps de `query()` entier
```python
    def query(
        self,
        question: str,
        use_query_transform: bool = False,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        if use_query_transform:
            logger.warning("use_query_transform=True non supporté en mode LangGraph — ignoré.")

        final_state = self._graph.invoke({
            "question": question,
            "history":  history or [],
        })

        elapsed = round(time.time() - start, 2)

        return {
            "question":        question,
            "search_query":    final_state.get("search_query", question),
            "answer":          final_state.get("answer", ""),
            "sources":         final_state.get("sources", []),
            "chunks_used":     final_state.get("chunks_used", 0),
            "elapsed_seconds": elapsed,
            "intent":          final_state.get("intent_data", {}),
            "warnings":        final_state.get("warnings", []),
        }
```

### 6e — Nettoyage (après validation)
Supprimer de `pipeline.py` (tout est déplacé dans `nodes.py`) :
- `_SYSTEM_PROMPT`, `_GENERATION_PROMPT`, `_GENERATION_PROMPT_LIST`
- Méthodes d'instance : `_format_history`, `_format_context`, `_extract_sources`, `_filter_by_source`, `_extract_primary_value`, `_fold`
- Méthodes uniquement utilisées dans `query()` : `_llm_validate_batch`, `_build_validation_prompt`, `_parse_kept_indices`, `_structured_query`
- **Garder** : `_deduplicate_chunks`, `_build_combined_schema`, `_normalize_stem` (utilisées par `ingest`)

---

## Ordre d'implémentation

1. `requirements.txt` — ajouter `langgraph`
2. `src/graph/__init__.py` — fichier vide
3. `src/graph/state.py` — TypedDict
4. `src/graph/nodes.py` — helpers + prompts + `build_nodes()` + `route_node()`
5. `src/graph/graph.py` — `build_rag_graph()`
6. `src/pipeline.py` — import + `_build_graph()` + `__init__` + `ingest`×2 + `query()` + nettoyage

---

## Points de vigilance

| Risque | Mitigation |
|--------|-----------|
| Closures périmées après `ingest()` | Reconstruire `self._graph = self._build_graph()` dans `ingest()` ET `ingest_documents()` |
| `exhaustive_node` retourne liste vide | Guard : retourner `{"answer": "Aucun résultat.", ...}` si `direct` est vide |
| `stream=True` non supporté | Log warning, `api.py` utilise toujours `stream=False` |
| `use_query_transform=True` non câblé | Log warning, idem non utilisé en prod |
| Import circulaire dans `nodes.py` | `reciprocal_rank_fusion` importé localement dans `retrieve_node` |

---

## Vérification

```bash
# 1. Import sanity (sans charger les modèles)
cd pfe26
python -c "from src.graph.graph import build_rag_graph; print('OK')"

# 2. Test routing logique
python -c "
from src.graph.nodes import route_node
from unittest.mock import MagicMock
schema = {'EMPLOYES': {'is_doc': False}}
s = MagicMock(); s.has_table.return_value = True
print(route_node({'intent_data': {'exhaustive': True,  'source': 'EMPLOYES', 'filter': None}}, schema, s))
print(route_node({'intent_data': {'exhaustive': False, 'source': 'EMPLOYES', 'filter': {'NOM': 'X'}}}, schema, s))
print(route_node({'intent_data': {'exhaustive': False, 'source': None,       'filter': None}}, schema, s))
"

# 3. Démarrer le serveur et tester l'endpoint /query normalement
```

La réponse JSON de `/query` reste identique — `api.py` et le frontend ne changent pas.
