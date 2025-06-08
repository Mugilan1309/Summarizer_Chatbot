import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def clean_input(self, texts):
        cleaned = []
        for t in texts:
            if isinstance(t, str) and t.strip():
                cleaned.append(t.strip())
            else:
                print("⚠️ Skipping invalid input to embedder:", t)
        return cleaned if cleaned else ["placeholder"]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        texts = self.clean_input(texts)
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, device=self.device)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def get_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
