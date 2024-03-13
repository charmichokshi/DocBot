from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_pdf_names(pdf_docs):
    pdf_names = ""
    for pdf in pdf_docs:
        pdf_names += pdf.name + " "
    return pdf_names
