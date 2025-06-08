from transformers import pipeline

def load_summarizer_model():
    return pipeline("summarization", model="google/pegasus-cnn_dailymail", device=-1)