import cohere
import langchain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings import CohereEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatCohere
from langchain.storage import LocalFileStore
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.cache import InMemoryCache


langchain.llm_cache = InMemoryCache()


def get_bm25_retriever(text_chunks):
    bm25_retriever = BM25Retriever.from_texts(text_chunks)
    bm25_retriever.k = 5
    return bm25_retriever


def get_cached_vector_store(_config, text_chunks, api_key):
    store = LocalFileStore("./cache/")
    embedding_model = CohereEmbeddings(cohere_api_key=api_key,
                                       model=_config.cohere_embedding_model)
    embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model,
                                                      store,
                                                      namespace=_config.cohere_embedding_model)
    vector_store = FAISS.from_texts(text_chunks, embedder)
    vector_store.save_local("faiss_index")


def RAG_hybrid(_config, api_key, ensemble_retriever):
    prompt_template = _config.llm_system_role
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatCohere(cohere_api_key=api_key)

    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=ensemble_retriever,
        verbose=True,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_with_sources_chain


def process_user_input_cache_hybrid(_config, user_question, api_key):
    co = cohere.Client(api_key)
    embeddings = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    query_emb = co.embed([user_question],
                         input_type="search_query",
                         model=_config.cohere_embedding_model).embeddings
    docs = new_db.similarity_search_by_vector(query_emb[0], k=5)  # 1024 dim emb vector

    bm25_retriever = get_bm25_retriever(user_question)
    faiss_retriever = new_db.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                           weights=[0.5, 0.5])

    chain = RAG_hybrid(_config, api_key, ensemble_retriever)
    response = chain({"input_documents": docs,
                      "query": user_question+_config.llm_format_output},
                       return_only_outputs=True,
                     )
    return response["result"]
