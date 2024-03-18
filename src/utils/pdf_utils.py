from PyPDF2 import PdfReader
import os


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def get_pdf_objects(pdf_files):
    documents = []

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_content = page.extract_text()

            metadata = {
                'source': pdf_file,
                'page': page_num
            }

            document = Document(page_content, metadata)
            documents.append(document)

    return documents


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
