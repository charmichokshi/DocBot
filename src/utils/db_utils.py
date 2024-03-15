import chromadb


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def get_chroma_client():
    chroma_client = chromadb.PersistentClient()
    return chroma_client
