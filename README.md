# DocBot
LLMs suffer from issues such as Knowledge cut-offs and hallucinations. They are not exposed to private data and lag the capability of being a fact engine. 

Retrieval-Augmented Generation (RAG) framework can solve these issues. This chatbot is built leveraging the free Cohere Model and Embeddings APIs. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience. (It will not hallucinate an answer if it's missing context.)

Below is a screenshot of how the App works.
![Streamlit App](https://github.com/charmichokshi/DocBot/blob/main/DocBot_SS.png)

### Technical Overview
- PDF Processing: Utilizes PyPDF2 for extracting text from PDF documents.
- Text Chunking: Employs the RecursiveCharacterTextSplitter from LangChain for dividing the extracted text into manageable chunks.
- Embeddings: Utilizes SOTA "embed-english-v3.0" model from CohereEmbeddings to generate embeddings of context documents and user queries.
- Vector Store Creation: Uses FAISS for creating a searchable vector store from text chunks.
- Answer Generation: Leverages ChatCohere from Cohere for generating answers to user queries using the context provided by the uploaded documents.

### To Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run <path_to_streamlit_app.py>`
