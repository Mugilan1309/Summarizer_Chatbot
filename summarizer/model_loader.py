from transformers import pipeline

def load_summarizer_model():
    return pipeline("summarization", model="t5-small")
