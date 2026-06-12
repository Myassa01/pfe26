# Myassa — Assistant RH IA (ENGTP / Sonatrach)

Système RAG 100% local pour la gestion des ressources humaines. Aucune clé API, aucune donnée envoyée à l'extérieur.

---

## Stack technique

| Composant | Technologie |
|---|---|
| LLM | HuggingFace Transformers — `Qwen/Qwen2.5-3B-Instruct` (local) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (local) |
| Vector DB | ChromaDB (persistant) |
| Recherche lexicale | BM25 (index JSON) |
| Données structurées | DuckDB — fichiers Excel (.xlsx) |
| Pipeline RAG | LangGraph (graphe d'états) |
| API | FastAPI — port `8001` |
| Frontend | React + Vite |
| Auth | JWT (TTL 8h) + SQLite |

---

## Prérequis

- Python 3.10+
- Node.js 18+
- ~4 Go RAM libre (modèles chargés en mémoire)
- GPU CUDA recommandé (fonctionne aussi sur CPU)

---

## Installation

### Backend

```bash
cd pfe26

# Créer et activer le venv
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

> Le premier lancement télécharge automatiquement les modèles HuggingFace (~1 Go au total). Une seule fois.

### Frontend

```bash
cd frontend
npm install
```

---

## Lancement

### Backend

```bash
cd pfe26
python api.py
```

L'API démarre sur `http://localhost:8001`.  
Documentation interactive : `http://localhost:8001/docs`

### Frontend

```bash
cd frontend
npm run dev
```

Interface disponible sur `http://localhost:5173`.

---

## Compte par défaut

| Champ | Valeur |
|---|---|
| Email | `admin@sonatrach.dz` |
| Mot de passe | `Admin@1234` |
| Rôle | `superadmin` |

---

## Utilisation

### 1. Ajouter des documents

Dépose tes fichiers dans `pfe26/documents/`. Formats supportés :

```
documents/
├── reglement_interieur.pdf
├── DEPARTEMENT.xlsx
├── DIRECTION.xlsx
├── guide_rh.docx
└── ...
```

Formats acceptés : `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.xlsx`, `.xls`

### 2. Ingérer les documents (interface admin)

Dans l'onglet **Documents** de l'interface, clique sur **Ingérer**.  
L'ingestion tourne en arrière-plan — surveille la progression via `GET /stats`.

Ou via l'API directement :

```bash
# Uploader un fichier
curl -X POST http://localhost:8001/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@mon_fichier.pdf"

# Lancer l'ingestion
curl -X POST http://localhost:8001/ingest \
  -H "Authorization: Bearer <token>"
```

### 3. Poser une question

Utilise l'onglet **Chat** de l'interface.

---

## Architecture du pipeline RAG (LangGraph)

Le pipeline de requête est un graphe d'états LangGraph avec 3 chemins de traitement :

```
START
  └─► contextualize   (résout les références implicites via l'historique)
        └─► intent     (classifie : list | detail | qa)
              └─► [routage conditionnel]
                    ├─► exhaustive    → finalize → END
                    │   (liste SQL directe, sans LLM)
                    ├─► structured_qa → finalize → END
                    │   (SQL + LLM, questions ciblées sur Excel)
                    └─► retrieve → rerank → generate → finalize → END
                        (RAG sémantique : hybride + reranking + LLM)
```

### Nœuds du graphe

| Nœud | Rôle |
|---|---|
| `contextualize` | Reformule la question si elle contient des références implicites ("ces", "leur", "eux"...) en utilisant l'historique de conversation |
| `intent` | Classifie l'intent (list / detail / qa), identifie la source et les filtres |
| `exhaustive` | Requête SQL DuckDB directe sur Excel — retourne une liste numérotée sans passer par le LLM |
| `structured_qa` | Requête SQL + génération LLM pour les questions ciblées avec filtre |
| `retrieve` | Recherche hybride : dense (ChromaDB) + sparse (BM25) fusionnés par RRF |
| `rerank` | Cross-encoder pour re-scorer et filtrer les chunks par source |
| `generate` | Génération LLM avec contexte + historique |
| `finalize` | Consolide sources et métadonnées pour la réponse finale |

---

## Structure du projet

```
pfe26/
├── api.py                  ← API REST FastAPI (port 8001)
├── auth.py                 ← JWT + gestion utilisateurs + historique (SQLite)
├── config.py               ← Paramètres (modèles, chunk size, chemins...)
├── requirements.txt        ← Dépendances Python
├── cv_analyzer.py          ← Analyse de CV (superadmin)
├── documents/              ← Fichiers à indexer (.pdf, .xlsx, .docx...)
├── data/                   ← Généré automatiquement
│   ├── users.db            ← Base SQLite (users, sessions, historique)
│   ├── chroma_db/          ← Index vectoriel ChromaDB
│   ├── bm25_index.json     ← Index BM25 (JSON, pas pickle)
│   └── links.json          ← URLs scrapées
└── src/
    ├── pipeline.py         ← RAGPipeline : init + ingest + query (via LangGraph)
    ├── graph/              ← Graphe LangGraph
    │   ├── state.py        ← GraphState TypedDict
    │   ├── nodes.py        ← Nœuds + helpers + prompts
    │   └── graph.py        ← build_rag_graph()
    ├── ingestion/          ← Chargement, chunking, embeddings
    ├── retrieval/          ← ChromaDB, BM25, fusion RRF
    ├── reranking/          ← Cross-encoder
    ├── generation/         ← LLM (HFClient), IntentRouter, QueryTransformer
    └── structured/         ← DuckDB — requêtes Excel

frontend/
├── src/
│   ├── api.js              ← Client HTTP (tous les endpoints)
│   ├── App.jsx             ← Routage + rôles
│   └── components/
│       ├── Chat.jsx        ← Interface de chat + historique
│       ├── Documents.jsx   ← Upload + ingestion
│       ├── Users.jsx       ← Gestion utilisateurs (superadmin)
│       └── Liens.jsx       ← Gestion URLs
└── package.json
```

---

## API — Endpoints principaux

Tous les endpoints protégés nécessitent : `Authorization: Bearer <token>`

### Authentification

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/auth/login` | Connexion → retourne le token JWT |
| `POST` | `/auth/logout` | Révocation du token |
| `GET` | `/auth/me` | Profil utilisateur connecté |

### Chat

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/query` | Poser une question |
| `GET` | `/historique` | Historique de l'utilisateur connecté |
| `DELETE` | `/historique` | Supprimer son historique |

**Corps de `/query` :**
```json
{
  "question": "Qui est le chef du département marketing ?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

**Réponse de `/query` :**
```json
{
  "question": "Qui est le chef du département marketing ?",
  "search_query": "chef département marketing",
  "answer": "MOHAND OUIDIR Chikhi est CHEF DE DEPARTEMENT MARKETING.",
  "sources": ["DEPARTEMENT.xlsx"],
  "chunks_used": 5,
  "elapsed_seconds": 4.2,
  "intent": { "intent": "qa", "source": "DEPARTEMENT", "exhaustive": false },
  "warnings": []
}
```

### Documents (superadmin)

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/upload` | Uploader un fichier |
| `POST` | `/ingest` | Lancer l'ingestion en arrière-plan |
| `POST` | `/reset` | Vider l'index |

### Liens (superadmin)

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/lien` | Ajouter des URLs et les indexer |
| `GET` | `/lien` | Lister les URLs enregistrées |
| `POST` | `/lien/scrape` | Re-scraper tous les liens stockés |

### Système

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Statut + nombre de chunks indexés |
| `GET` | `/stats` | Configuration complète + statut ingestion |

### Utilisateurs (superadmin)

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/users` | Liste des utilisateurs |
| `POST` | `/users` | Créer un utilisateur |
| `PUT` | `/users/{id}` | Modifier un utilisateur |
| `DELETE` | `/users/{id}` | Désactiver un utilisateur |

---

## Configuration

Paramètres dans [config.py](config.py) :

| Paramètre | Défaut | Description |
|---|---|---|
| `llm_model` | `Qwen/Qwen2.5-3B-Instruct` | Modèle LLM HuggingFace |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embedding |
| `reranker_model` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Modèle de reranking |
| `chunk_size` | `256` | Taille des chunks en tokens |
| `chunk_overlap` | `32` | Chevauchement entre chunks |
| `top_k_dense` | `20` | Résultats dense avant RRF |
| `top_k_sparse` | `20` | Résultats BM25 avant RRF |
| `top_k_after_rerank` | `5` | Chunks finaux envoyés au LLM |
| `rrf_k` | `60` | Paramètre de fusion RRF |
| `llm_max_tokens` | `512` | Longueur max de la réponse |

---

## Rôles utilisateurs

| Rôle | Accès |
|---|---|
| `employee` | Chat, historique personnel |
| `superadmin` | Tout + gestion documents, liens, utilisateurs, analyse CV |

---

## Gestion des erreurs API

| Code | Signification |
|---|---|
| `400` | Question vide ou paramètre invalide |
| `401` | Token manquant ou expiré |
| `403` | Accès réservé aux superadmins |
| `409` | Ingestion déjà en cours |
| `500` | Erreur pipeline interne |
