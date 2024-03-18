import streamlit as st
from utils.load_config import LoadConfig
from utils.app_utils import get_text_chunks, get_vector_store, process_user_input, get_cached_vector_store, \
    get_doc_chunks, get_chroma_vector_store
from utils.pdf_utils import get_pdf_text, delete_faiss_files, get_pdf_objects
import time

APPCFG = LoadConfig()
cohere_api_key = st.secrets["COHERE_API_KEY"]

st.set_page_config(page_title="DocBot", layout="wide")


def main():
    # delete_faiss_files('./faiss_index/')
    st.markdown("""## DocBot ⛵: Chat with PDFs and get instant insights!""")

    with st.expander("About"):
        st.markdown("""
        This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Cohere Model and Embeddings.
        It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. 
        This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience. 
        (It will not hallucinate an answer if its missing context.)
        """)

    # How It Works section
    with st.expander("How It Works"):
        st.markdown("""
        1. **Upload Your Documents**: You can upload multiple PDF files at once. Press Submit & Process.
        2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
        """)
    # st.divider()

    with st.sidebar:
        st.title("Uploader:")
        pdf_docs = st.file_uploader("Upload PDF Files, Configure options and Click on the Submit & Process Button",
                                    accept_multiple_files=True, key="pdf_uploader")
        storage_type = st.radio("Select Storage Type", ["Chroma", "FAISS"], index=0)
        if storage_type == "FAISS":
            cache_or_history = st.radio("Select Advanced Features",
                                        ["Cache Embeddings (IM)", "Conversational System",
                                         "Hybrid Search + Cache (P)", "None"], index=3)
        else:
            cache_or_history = False

        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                # FAISS CACHE
                if storage_type == "FAISS" and cache_or_history == "Cache Embeddings":
                    raw_docs = get_pdf_objects(pdf_docs)
                    docs_chunks = get_doc_chunks(APPCFG, raw_docs)
                    get_cached_vector_store(APPCFG, docs_chunks, cohere_api_key)
                    st.success("Done!!")
                    # st.write('Using FAISS with Cache...')
                # FAISS W/O CACHE
                elif storage_type == "FAISS":
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(APPCFG, raw_text)
                    get_vector_store(APPCFG, text_chunks, cohere_api_key)
                    st.success("Done!!")
                    # st.write('Using FAISS without Cache...')
                # CHROMA W/O CACHE
                elif storage_type == "Chroma":
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(APPCFG, raw_text)
                    get_chroma_vector_store(APPCFG, text_chunks, cohere_api_key)
                    st.success("Done!!")
                    # st.write('Using Chroma DB...')

    st.markdown(
        """
        <style>
        .stTextInput > div:last-child {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if pdf_docs:
        user_question = st.text_input(r"$\textsf{\Large Start Chatting 💬}$",
                                      placeholder="Type your question here and hit Ask",
                                      disabled=False)
        submitted = st.button("Ask ➤", disabled=False)
    else:
        user_question = st.text_input(r"$\textsf{\Large Start Chatting 💬}$",
                                      placeholder="Type your question here and hit Ask",
                                      disabled=True)
        submitted = st.button("Ask ➤", disabled=True,
                              help="Please upload PDF(s) and enter a question before submitting.")

    use_history = False
    if cache_or_history == "Conversational System":
        use_history = True

    hybrid_search = False
    if cache_or_history == "Hybrid Search + Cache":
        hybrid_search = True

    if st.session_state.get("submitted") or submitted:
        if user_question:
            start_time = time.time()
            response = process_user_input(APPCFG, user_question, cohere_api_key,
                                          storage_type, use_history, hybrid_search)
            end_time = time.time()
            st.write("DocBot ⛵: ", response)
            execution_time = end_time - start_time
            st.write("Execution time:", round(execution_time, 2), "seconds")
        else:
            st.error("Please enter a question before submitting.")
        st.session_state["submitted"] = False


if __name__ == "__main__":
    main()
