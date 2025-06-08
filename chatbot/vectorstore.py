import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim: int):
        """
        Initializes a FAISS index for fast similarity search.
        Uses Inner Product (IP) for cosine similarity search (assuming normalized vectors).
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Use IP for cosine similarity
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, texts: list[str]):
        """
        Adds embeddings and their corresponding text chunks to the store.
        """
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [(self.text_chunks[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
