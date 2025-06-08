import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text") + "\n\n"
    return full_text