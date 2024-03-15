import chromadb


def get_chroma_client():
    chroma_client = chromadb.PersistentClient()
    return chroma_client
