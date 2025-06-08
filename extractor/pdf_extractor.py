import fitz  # PyMuPDF
from io import BytesIO

def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    pdf_buffer = BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_buffer, filetype="pdf")

    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text") + "\n\n"
    return full_text
