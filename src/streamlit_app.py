import streamlit as st
from utils.load_config import LoadConfig
from utils.app_utils import get_text_chunks, get_vector_store, user_input
from utils.pdf_utils import get_pdf_text


APPCFG = LoadConfig()
cohere_api_key = APPCFG.load_cohere_credentials()

st.set_page_config(page_title="DocBot", layout="wide")
st.markdown("""
## DocBot ⛵: Chat with PDFs and get instant insights!

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Cohere Model and Embeddings.
It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. 
This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience. 
(It will not hallucinate an answer if its missing context.)""")
# st.divider()
st.markdown("""
##### How It Works

1. **Upload Your Documents**: You can upload multiple PDF files at once. Press Submit & Process.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")
st.divider()


def main():
    st.header("Start Chatting 💬")

    # user_question = st.chat_input("Ask a Question from the PDF Files", key="user_question")
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        response = user_input(APPCFG, user_question, cohere_api_key)
        st.write("DocBot ⛵: ", response)

    with st.sidebar:
        st.title("Uploader:")
        pdf_docs = st.file_uploader("Upload PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(APPCFG, raw_text)
                get_vector_store(APPCFG, text_chunks, cohere_api_key)
                st.success("Done!!")


if __name__ == "__main__":
    main()
