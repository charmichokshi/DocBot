from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import CohereEmbeddings
from langchain.chat_models import ChatCohere
from .pdf_utils import get_pdf_names

def get_text_chunks(_config, text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_config.chunk_size,
                                                   chunk_overlap=_config.chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(_config, text_chunks, api_key):
    embeddings = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def RAG(_config, api_key):
    prompt_template = _config.llm_system_role
    model = ChatCohere(cohere_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(_config, user_question, api_key):
    embeddings = CohereEmbeddings(cohere_api_key=api_key, model=_config.cohere_embedding_model)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = RAG(_config, api_key)
    response = chain({"input_documents": docs,
                      "question": user_question},
                     return_only_outputs=True)
    return response["output_text"]

