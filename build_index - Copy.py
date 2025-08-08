import faiss
import numpy as np
import pickle
from typing import List, Optional
import os

class SemanticSearchIndex:
    def __init__(self):
        self.index = None
        self.texts = []
        self.embedding_dim = None

    def build(self, embeddings: List[List[float]], texts: List[str]) -> None:
        """Build the FAISS index"""
        if not embeddings or not texts:
            raise ValueError("Empty embeddings or texts provided")
            
        if len(embeddings) != len(texts):
            raise ValueError("Mismatch between embeddings and texts length")

        self.embedding_dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts = texts

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Search the index"""
        if self.index is None:
            raise RuntimeError("Index not built - call build() first")
            
        if len(query_embedding) != self.embedding_dim:
            raise ValueError("Query embedding dimension mismatch")

        vectors = np.array([query_embedding]).astype("float32")
        _, indices = self.index.search(vectors, top_k)
        
        return [self.texts[idx] for idx in indices[0] if 0 <= idx < len(self.texts)]

    def save(self, index_path: str = "faiss.index", text_path: str = "texts.pkl") -> None:
        """Save index to disk"""
        if self.index is None:
            raise RuntimeError("Index not built - nothing to save")
            
        faiss.write_index(self.index, index_path)
        with open(text_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, index_path: str = "faiss.index", text_path: str = "texts.pkl") -> None:
        """Load index from disk"""
        if not os.path.exists(index_path) or not os.path.exists(text_path):
            raise FileNotFoundError("Index files not found")
            
        self.index = faiss.read_index(index_path)
        with open(text_path, "rb") as f:
            self.texts = pickle.load(f)
        self.embedding_dim = self.index.d