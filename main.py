import streamlit as st

if "agreed_to_terms" not in st.session_state:
    st.session_state.agreed_to_terms = False

if not st.session_state.agreed_to_terms:
    st.warning("⚠️ This is a prototype. Summaries and chatbot responses may be inaccurate. Do not rely on this information for real academic or legal decisions.")
    agree = st.button("I Understand & Agree")
    if agree:
        st.session_state.agreed_to_terms = True
    else:
        st.stop()

from extractor.pdf_extractor import extract_text_from_pdf
from summarizer.model_loader import load_summarizer_model
from summarizer.summarization import summarize_document
from chatbot.chatbot_runner import ChatbotRunner
from chatbot.embedder import Embedder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

RELEVANCE_THRESHOLD = 0.30
TOP_K_RELEVANT_CHUNKS = 5

class RelevanceFilter:
    def __init__(self, embedder, chunk_embeddings, threshold=0.30, top_k=5):
        self.embedder = embedder
        self.chunk_embeddings = chunk_embeddings
        self.threshold = threshold
        self.top_k = top_k

    def get_relevant_chunks(self, query: str):
        query_embedding = self.embedder.embed_query(query)
        scores = cosine_similarity(query_embedding.reshape(1, -1), self.chunk_embeddings).flatten()
        relevant_indices = np.where(scores >= self.threshold)[0]
        sorted_indices = relevant_indices[np.argsort(scores[relevant_indices])[::-1]]
        top_indices = sorted_indices[:self.top_k]
        return top_indices, scores[top_indices]

def chunk_text_with_overlap(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (max_words - overlap)
    return chunks

# Streamlit UI
st.set_page_config(page_title="📄 Summarizer + Chatbot", layout="wide")
st.title("📄 Research Paper Summarizer + Document Chatbot")

st.markdown("---")
st.markdown(
    "<small>⚠️ This is a prototype built with limited compute. Summaries and chatbot answers may be inaccurate or incomplete. Do not rely on this for academic, legal, or medical decisions.</small>",
        unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Cache models to avoid reloads
    @st.cache_resource
    def load_models():
        return load_summarizer_model(), ChatbotRunner(), Embedder()

    with st.spinner("Loading models..."):
        summarizer_model, chatbot, embedder = load_models()

    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(uploaded_file).replace("<n>", "\n")

    st.success(f"✅ Extracted {len(full_text.split())} words from the document.")

    # Initialize session state variables
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chunks" not in st.session_state:
        with st.spinner("Chunking & indexing document..."):
            chunks = chunk_text_with_overlap(full_text)
            chunk_embeddings = embedder.embed_texts(chunks)
            st.session_state.chunks = chunks
            st.session_state.chunk_embeddings = chunk_embeddings
            st.session_state.relevance_filter = RelevanceFilter(
                embedder, chunk_embeddings, RELEVANCE_THRESHOLD, TOP_K_RELEVANT_CHUNKS
            )
            chatbot.index_document(chunks)
        st.success("Chatbot ready!")
    else:
        chunks = st.session_state.chunks
        relevance_filter = st.session_state.relevance_filter

    # Summary generation
    if st.button("🔍 Generate Summary") and st.session_state.summary is None:
        with st.spinner("Summarizing document..."):
            summary = summarize_document(full_text, summarizer_model, uploaded_file.getvalue())
            st.session_state.summary = summary.replace("<n>", "\n")

    if st.session_state.summary:
        st.subheader("📝 Summary")
        st.write(st.session_state.summary)

    st.divider()
    st.subheader("💬 Ask Questions About the PDF")

    query = st.text_input("Ask a question:")
    if query:
        relevance_filter = st.session_state.relevance_filter
        top_indices, scores = relevance_filter.get_relevant_chunks(query)
        if len(top_indices) == 0:
            st.error("❌ Sorry, this question doesn't seem to relate to the document.")
        else:
            relevant_chunks = [chunks[i] for i in top_indices]
            chatbot_context = "\n".join(relevant_chunks)

            with st.spinner("Generating answer..."):
                response = chatbot.generate_response(
                    query,
                    context_override=chatbot_context,
                    chat_history=st.session_state.chat_history
                )

            st.session_state.chat_history.append((query, response))

            st.markdown("### Conversation so far:")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")

            st.markdown("### Latest Answer:")
            st.write(response)
