# DocBot ⛵
LLMs suffer from issues such as Knowledge cut-offs and hallucinations. They are not exposed to private data and lag the capability of being a fact engine. 

Retrieval-Augmented Generation (RAG) framework can solve these issues. This chatbot is built leveraging the free Cohere Model and Embeddings APIs. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience. (It will not hallucinate an answer if it's missing context.)

Below is a screenshot of how the App works.
![Streamlit App](https://github.com/charmichokshi/DocBot/blob/main/images/DocBot_SS.png)

### Technical Overview

- **PDF Processing:** Utilizes PyPDF2 for extracting text from PDF documents.
- **Text Chunking:** Employs the RecursiveCharacterTextSplitter from LangChain for dividing the extracted text into manageable chunks.
- **Embeddings:** Utilizes SOTA `embed-english-v3.0` model from CohereEmbeddings to generate embeddings of context documents and user queries.
- **Vector Store Creation:** Uses FAISS for creating a searchable vector store from text chunks.
- **Answer Generation:** Leverages ChatCohere from Cohere for generating answers to user queries using the context provided by the uploaded documents.

**Advanced RAG Concepts Implemented**
- **Caching Embeddings:** Embeddings can be stored or temporarily cached to avoid needing to recompute them using `CacheBackedEmbeddings`
- **Hybrid vector Search:** It has the advantage of doing keyword search as well as the advantage of doing a semantic lookup that we get from embeddings and a vector search. I have used FAISS for semantic search and BM25 Algorithm for keyword search to implementing Hybrid Search using `langchainEnsembleRetriever`.
- **InMemory Caching:** Caching happens for every user query. The response is generated for each user query if it does not match with previously requested queries.
  
### To Run

1. Clone this Repository
2. Create a Python Virtual Environment
3. Install requirements: `pip install -r requirements.txt`
4. Run the app: `streamlit run src/streamlit_app.py`
5. Open the 'External URL' in your browser and start Chatting ⛵
