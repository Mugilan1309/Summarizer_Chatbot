# SmartScholar: Research Paper Summarizer + Document Chatbot

A Streamlit-based prototype that allows users to upload research papers (PDFs), generate summaries, and ask questions about the document using a built-in chatbot.

> ⚠️ **Disclaimer**  
> This is a prototype. Summaries and chatbot responses may be inaccurate or incomplete. Do **not** rely on this application for academic, legal, medical, or other critical decisions.

---

## ✨ Features

- 📄 Upload any PDF document (preferably research papers)
- 📝 Generate concise summaries using an ML-based summarizer
- 💬 Ask context-aware questions about the uploaded PDF
- 🔍 Relevance filtering to limit chatbot responses to top relevant sections
- ⚠️ Agreement flow to warn users about limitations

---

## 🛠️ How It Works

- **PDF Extraction**: Extracts raw text from the uploaded PDF
- **Text Chunking**: Splits text into overlapping chunks for embedding
- **Embedding + Indexing**: Converts text chunks into embeddings and indexes them
- **Summarizer**: Generates a summary using a pretrained model
- **Chatbot**: Answers user queries by retrieving the most relevant chunks and responding based on them

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- pip (Python package manager)

### Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```
Run the App
```
streamlit run main.py
```
🧪 Project Structure
```
📁 Project Root
├── main.py                         # Streamlit app entry point
├── requirements.txt               # Python dependencies

├── extractor/
│   └── pdf_extractor.py           # Extracts text from uploaded PDFs

├── parser/
│   ├── chunker.py                 # Splits text into overlapping chunks
│   └── heading_detector.py        # Detects section headings in documents

├── summarizer/
│   ├── model_loader.py            # Loads summarization model
│   ├── summarization.py           # Logic for generating summaries
│   └── utils.py                   # Utility functions for summarization

├── chatbot/
│   ├── embedder.py                # Converts text to vector embeddings
│   ├── vectorstore.py             # Handles vector-based retrieval
│   ├── rag_pipeline.py            # Retrieval-Augmented Generation pipeline
│   └── chatbot_runner.py          # Main chatbot logic
```
## 📌 Usage Notes
- A user must agree to a disclaimer before using the app.
- The chatbot uses cosine similarity to match questions to the most relevant chunks of the PDF.
- Summarizer may take time depending on file length and system resources.
- Models are cached to improve performance on repeated runs.

## 🙏 Credits
- Streamlit
- Hugging Face Transformers
- scikit-learn
- PyMuPDF / pdfplumber / your PDF lib
- NumPy

## 📣 Disclaimer
This is a non-commercial, experimental project for learning and demonstration purposes only. The generated outputs (summaries and chatbot responses) are not guaranteed to be accurate or complete.
