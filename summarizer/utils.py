import fitz  # PyMuPDF

def detect_image_only_pages(pdf_path, word_threshold=20):
    """
    Detect pages in the PDF that have very few or no text blocks, likely containing only images.
    Returns a list of (page_number, is_image_only) tuples.
    """
    doc = fitz.open(pdf_path)
    image_only_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        text_block_count = 0
        word_count = 0

        for block in blocks:
            if block["type"] == 0:  # text block
                text = "".join([line["spans"][0]["text"] for line in block["lines"] if line["spans"]])
                word_count += len(text.split())
                text_block_count += 1

        if word_count < word_threshold:
            image_only_pages.append(page_num)

    return image_only_pages

def is_section_image_only(start_pos, end_pos, headings_positions, image_only_pages, doc):
    """
    Determine if a section (from start_pos to end_pos in text) corresponds mainly to image-only pages.
    This requires mapping text offsets to page numbers roughly.
    """

    # Map text positions to page numbers roughly by cumulative text length per page
    page_text_lengths = []
    cumulative = 0
    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text("text")
        cumulative += len(page_text)
        page_text_lengths.append(cumulative)

    # Find start page number
    start_page = 0
    for i, cum_len in enumerate(page_text_lengths):
        if start_pos < cum_len:
            start_page = i
            break

    # Find end page number
    end_page = start_page
    for i, cum_len in enumerate(page_text_lengths[start_page:], start=start_page):
        if end_pos < cum_len:
            end_page = i
            break
    else:
        end_page = len(doc) - 1

    # Check if majority pages in section are image-only
    section_pages = set(range(start_page, end_page + 1))
    image_pages_in_section = section_pages.intersection(image_only_pages)

    # If more than half of pages in section are image-only, return True
    if len(image_pages_in_section) >= len(section_pages) / 2:
        return True
    return False
