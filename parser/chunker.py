def chunk_large_section(text, max_words=400):
    """Split a large section into smaller chunks by paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_len + para_words > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_len = para_words
        else:
            current_chunk.append(para)
            current_len += para_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
