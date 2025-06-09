from transformers import pipeline, set_seed, AutoTokenizer
from chatbot.rag_pipeline import RAGPipeline

class ChatbotRunner:
    def __init__(self, embedder_model_name="sentence-transformers/all-MiniLM-L6-v2", llm_model_name="google/flan-t5-small"):
        self.rag = RAGPipeline(embedder_model_name=embedder_model_name)
        self.generator = pipeline("text2text-generation", model=llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        set_seed(42)

    def index_document(self, text_chunks: list[str]):
        self.rag.index_document(text_chunks)

    def generate_response(self, query: str, max_length=256, top_k=5, context_override=None, chat_history=None) -> str:
        if context_override is None:
            context_chunks = self.rag.retrieve_context(query, top_k=top_k)
        else:
            context_chunks = context_override.split("\n")

        context_text = "\n".join(context_chunks)
        chat_history = chat_history or []
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])

        prompt = (
            f"Answer the question based only on the following document context. "
            f"If the answer is not contained in the context, say \"I don't know.\"\n\n"
            f"Context:\n{context_text}\n\n"
            f"Chat history:\n{history_text}\n\n"
            f"Q: {query}\nA:"
        )

        input_ids = self.tokenizer.encode(prompt, truncation=True, max_length=512)
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        outputs = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=False,  
            repetition_penalty=1.1
        )

        return outputs[0]['generated_text'].strip()
