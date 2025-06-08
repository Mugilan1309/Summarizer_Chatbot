import logging
import re
import fitz  # PyMuPDF

from parser.heading_detector import detect_headings_from_lines, chunk_text_by_headings
from parser.chunker import chunk_large_section
from summarizer.utils import detect_image_only_pages, is_section_image_only


def extract_lines_with_bold_info(pdf_path):
    doc = fitz.open(pdf_path)
    text_lines = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                line_text = "".join(span["text"] for span in spans).strip()
                is_bold = any("bold" in span["font"].lower() or "demibold" in span["font"].lower() for span in spans)
                y0 = line["bbox"][1]
                if line_text:
                    text_lines.append((page_num, y0, line_text, is_bold))

    # Sort by page and vertical position
    text_lines.sort(key=lambda x: (x[0], x[1]))
    return text_lines


def summarize_section(text, summarizer, max_words=400):
    chunks = chunk_large_section(text, max_words=max_words)
    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f" Summarizing chunk {i}/{len(chunks)} (words: {len(chunk.split())})...")
        summary = summarizer(chunk, min_length=30, max_length=100)[0]['summary_text']
        chunk_summaries.append(summary)

    return " ".join(chunk_summaries)


def polish_summary(summary: str) -> str:
    # Remove broken <n> artifacts
    summary = summary.replace("<n>", "\n")

    # Fix extra spaces around punctuation
    summary = re.sub(r'\s+([.,!?;:])', r'\1', summary)

    # Capitalize sentence starts
    summary = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), summary)
    summary = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), summary)  # Start of string

    # Fix common artifacts
    summary = re.sub(r'\bes related\b', 'issues related', summary)
    summary = re.sub(r'\bth\.\b', 'the.', summary)

    # Remove filler phrases
    summary = re.sub(r'\bthe purpose of this study was to examine\b.*?\.', '', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\bthis paper (focuses|discusses)\b', 'The study \\1', summary, flags=re.IGNORECASE)

    # Remove extra newlines
    summary = re.sub(r'\n{2,}', '\n', summary)

    # Deduplicate sentences
    seen = set()
    lines = []
    for sentence in summary.split('. '):
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            lines.append(sentence)
    summary = '. '.join(lines)

    # Ensure ending period
    if not summary.endswith('.'):
        summary += '.'

    return summary


def summarize_document(text, summarizer, pdf_path):
    """
    Summarize a large text chunked by headings, skipping image-only or empty sections.

    Args:
        text (str): Full text of the document.
        summarizer (callable): Function that summarizes a string input.
        pdf_path (str): Path to the PDF file (used to detect image-only pages).

    Returns:
        str: Polished concatenated summary of all text sections.
    """
    logging.basicConfig(level=logging.INFO)
    doc = fitz.open(pdf_path)
    image_only_pages = detect_image_only_pages(pdf_path)

    text_lines = extract_lines_with_bold_info(pdf_path)
    headings = detect_headings_from_lines(text_lines)
    sections = chunk_text_by_headings(text, headings)

    final_summary = []
    last_pos = 0

    for heading, content in sections.items():
        content = content.strip()
        if not content:
            logging.warning(f"Section '{heading}' is empty or contains no extractable text. Skipping.")
            continue

        start_pos = text.find(content, last_pos)
        if start_pos == -1:
            logging.warning(f"Could not locate content of section '{heading}' in the main text. Skipping.")
            continue
        end_pos = start_pos + len(content)
        last_pos = end_pos

        if is_section_image_only(start_pos, end_pos, [], image_only_pages, doc):
            logging.warning(f"Section '{heading}' appears to contain mostly images. Skipping summarization.")
            final_summary.append(
                f"{heading.capitalize()}:\n⚠️ This section appears to contain mostly images and was skipped from summarization.\n\n"
            )
            continue

        logging.info(f"Summarizing section '{heading}'...")
        summary = summarize_section(content, summarizer)
        polished_summary = polish_summary(summary)
        final_summary.append(f"{heading.capitalize()}:\n{polished_summary}\n\n")  

    return "".join(final_summary).strip()

