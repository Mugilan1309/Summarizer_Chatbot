from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

def load_summarizer_model():
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail", use_fast=False)
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
    return pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
