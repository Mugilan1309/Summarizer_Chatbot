from chatbot.embedder import Embedder
from chatbot.vectorstore import VectorStore

class RAGPipeline:
    def __init__(self, embedder_model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = Embedder(model_name=embedder_model_name)
        self.vector_store = VectorStore(embedding_dim=self.embedder.get_dim())

    def index_document(self, chunks: list[str]):
        """
        Embeds and indexes the text chunks.
        """
        embeddings = self.embedder.embed_texts(chunks)
        self.vector_store.add_embeddings(embeddings, chunks)

    def retrieve_context(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieves top_k most relevant chunks for the user query.
        """
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.query(query_embedding, top_k=top_k)
