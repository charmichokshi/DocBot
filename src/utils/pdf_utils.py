from PyPDF2 import PdfReader
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        text += 'The File Name is ' + pdf.name + '.'
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        text += ' The File Name is ' + pdf.name + '.'
    return text


def delete_faiss_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file and delete it
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Failed to delete file: {file_path}. Error: {e}")
