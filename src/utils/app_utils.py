from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import CohereEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatCohere
from langchain.cache import InMemoryCache
from langchain.vectorstores.chroma import Chroma
from typing import Dict, List, Optional, Tuple


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


chroma_vector_store = None


class InMemoryCache:
    def __init__(self):
        self.cache: Dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self.cache.get(key)

    def set(self, key: str, value: bytes):
        self.cache[key] = value

    def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        return [self.cache.get(key) for key in keys]

    def mset(self, key_value_pairs: List[Tuple[str, bytes]]):
        for key, value in key_value_pairs:
            self.cache[key] = value

    def __len__(self):
        return len(self.cache)

    def __str__(self):
        return str(self.cache)


def get_doc_chunks(_config, raw_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_config.chunk_size,
                                                   chunk_overlap=_config.chunk_overlap)
    doc_chunks = text_splitter.transform_documents(raw_docs)
    return doc_chunks


def get_cached_vector_store(_config, docs_chunks, api_key):
    embedding_model = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
    store = InMemoryCache()
    embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model,
                                                      store,
                                                      namespace=_config.cohere_embedding_model)
    vector_store = FAISS.from_documents(docs_chunks, embedder)
    vector_store.save_local("faiss_index")


def get_text_chunks(_config, text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_config.chunk_size,
                                                   chunk_overlap=_config.chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(_config, text_chunks, api_key):
    embeddings = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_chroma_vector_store(_config, text_chunks, api_key):
    global chroma_vector_store
    embeddings = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    chroma_vector_store = vector_store


def RAG(_config, api_key):
    prompt_template = _config.llm_system_role
    model = ChatCohere(cohere_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model,
                          chain_type="stuff",
                          prompt=prompt)
    return chain


def process_user_input(_config, user_question, api_key, storage_type):
    if storage_type == "FAISS":
        embeddings = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    elif storage_type == "Chroma":
        global chroma_vector_store
        docs = chroma_vector_store.similarity_search(user_question)
    else:
        return ""

    chain = RAG(_config, api_key)
    response = chain({"input_documents": docs,
                      "question": user_question},
                     return_only_outputs=True)
    # "question": user_question+_config.llm_format_output},
    return response["output_text"]
