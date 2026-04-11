"""Interface ChromaDB pour le vector store local persistant."""
from typing import List, Dict, Any
import numpy as np


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store: {self.collection.count()} chunks indexés")

    def add(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Ajoute ou met à jour des documents dans le vector store."""
        # ChromaDB attend des listes de floats
        emb_list = [e.tolist() for e in embeddings]
        # Valeurs de metadata doivent être des primitives (str, int, float, bool)
        clean_meta = [
            {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            for m in metadatas
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=emb_list,
            documents=texts,
            metadatas=clean_meta,
        )

    def search(self, query_embedding: np.ndarray, k: int = 20) -> List[Dict[str, Any]]:
        """Recherche sémantique — retourne top-k résultats."""
        n = self.collection.count()
        if n == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # distance → similarity
                "rank": i + 1,
            })
        return docs

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Supprime tous les documents et recrée la collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store réinitialisé")
