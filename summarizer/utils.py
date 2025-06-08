import fitz  # PyMuPDF

def detect_image_only_pages(doc, word_threshold=20):
    """
    Detect pages in the PDF that have very few or no text blocks, likely containing only images.
    Returns a list of (page_number) for image-only pages.
    
    Args:
        doc (fitz.Document): An opened PyMuPDF document.
    """
    image_only_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        word_count = 0

        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        word_count += len(span["text"].split())

        if word_count < word_threshold:
            image_only_pages.append(page_num)

    return image_only_pages


def is_section_image_only(start_pos, end_pos, headings_positions, image_only_pages, doc):
    """
    Determine if a section (from start_pos to end_pos in text) corresponds mainly to image-only pages.
    
    Args:
        start_pos (int): Start character index of section in full text.
        end_pos (int): End character index of section.
        headings_positions (unused): Placeholder for possible heading position data.
        image_only_pages (List[int]): List of image-only page indices.
        doc (fitz.Document): The PyMuPDF document.
    
    Returns:
        bool: True if section is mostly image-only pages.
    """
    # Estimate text position -> page number by character count
    page_text_lengths = []
    cumulative = 0
    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text("text")
        cumulative += len(page_text)
        page_text_lengths.append(cumulative)

    # Find start page
    start_page = 0
    for i, cum_len in enumerate(page_text_lengths):
        if start_pos < cum_len:
            start_page = i
            break

    # Find end page
    end_page = start_page
    for i, cum_len in enumerate(page_text_lengths[start_page:], start=start_page):
        if end_pos < cum_len:
            end_page = i
            break
    else:
        end_page = len(doc) - 1

    section_pages = set(range(start_page, end_page + 1))
    image_pages_in_section = section_pages.intersection(image_only_pages)

    return len(image_pages_in_section) >= len(section_pages) / 2
