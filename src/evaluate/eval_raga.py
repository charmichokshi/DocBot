import os
import toml
import warnings
warnings.filterwarnings('ignore')

from pypdf import PdfReader
from utils.load_config import LoadConfig
from utils.pdf_utils import Document
from utils.app_utils import (
    get_text_chunks,
    get_vector_store,
    get_chroma_vector_store,
    process_user_input,
    chroma_vector_store
)
from utils.hybrid_app_utils import get_cached_vector_store as get_hybrid_vector_store

from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def main():
    # Load Cohere API Key from secrets
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets.get("COHERE_API_KEY")
    if not api_key or "your_cohere" in api_key:
        print("Error: Please set your actual COHERE_API_KEY in .streamlit/secrets.toml first.")
        return

    print("Loading PDF and extracting sample pages for evaluation...")
    reader = PdfReader("data/2023-annual-report-e.pdf")
    # Load pages 2 to 5 (0-indexed 1 to 4)
    raw_text = ""
    for page_num in range(1, 5):
        page_content = reader.pages[page_num].extract_text()
        raw_text += page_content

    config = LoadConfig()
    text_chunks = get_text_chunks(config, raw_text)

    # Define test dataset
    questions = [
        "Who is the Group President and CEO of TD Bank Group in 2023?",
        "What are the pillars of TD's strategy guided by their purpose?",
        "What is TD Insurance's ranking in the direct insurance market in Canada?"
    ]
    ground_truths = [
        "Bharat Masrani is the Group President and Chief Executive Officer of TD Bank Group in 2023.",
        "The pillars of TD's strategy are: Proven Business Model, Purpose-Driven, and Forward-Focused.",
        "TD Insurance is Canada's number-one direct insurer."
    ]

    # Helper function to get retrieved contexts for each question
    def get_retrieved_contexts(question, strategy, chunks=None):
        if strategy == "Chroma":
            import utils.app_utils as app_utils
            docs = app_utils.chroma_vector_store.similarity_search(question)
        elif strategy == "FAISS" or strategy == "FAISS_Conversational":
            embeddings = CohereEmbeddings(cohere_api_key=api_key, model=config.cohere_embedding_model)
            faiss_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = faiss_db.similarity_search(question)
        elif strategy == "FAISS_Hybrid":
            embeddings = CohereEmbeddings(cohere_api_key=api_key, model=config.cohere_embedding_model)
            faiss_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            bm25_retriever = BM25Retriever.from_texts(chunks)
            bm25_retriever.k = 5
            faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
            docs = ensemble_retriever.invoke(question)
        else:
            docs = []
        return [doc.page_content for doc in docs]

    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=40), reraise=True)
    def run_rag_with_retry(run_rag_fn, question):
        return run_rag_fn(question)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=40), reraise=True)
    def setup_with_retry(setup_fn):
        setup_fn()

    strategies = [
        {
            "name": "Chroma",
            "setup": lambda: get_chroma_vector_store(config, text_chunks, api_key),
            "run_rag": lambda q: process_user_input(config, q, api_key, "Chroma", False, False)
        },
        {
            "name": "FAISS",
            "setup": lambda: get_vector_store(config, text_chunks, api_key),
            "run_rag": lambda q: process_user_input(config, q, api_key, "FAISS", False, False)
        },
        {
            "name": "FAISS_Conversational",
            "setup": lambda: get_vector_store(config, text_chunks, api_key),
            "run_rag": lambda q: process_user_input(config, q, api_key, "FAISS", True, False)
        },
        {
            "name": "FAISS_Hybrid",
            "setup": lambda: get_hybrid_vector_store(config, text_chunks, api_key),
            "run_rag": lambda q: process_user_input(config, q, api_key, "FAISS", False, True)
        }
    ]

    # Setup wrapper models for Ragas evaluation using Cohere command-r (high quality)
    evaluator_llm = LangchainLLMWrapper(ChatCohere(cohere_api_key=api_key, max_tokens=2048), is_finished_parser=lambda x: True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(cohere_api_key=api_key, model=config.cohere_embedding_model))

    metrics = [faithfulness, answer_relevancy]
    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    # Stage 1: Generate answers and save immediate comparison table (100% reliable)
    print("\n--- Running RAG Strategies and Generating Answers ---")
    raw_results = []
    all_datasets = {}

    for strat in strategies:
        print(f"\nSetting up database and running Strategy: {strat['name']}")
        setup_with_retry(strat["setup"])
        
        answers = []
        contexts_list = []
        
        for idx, q in enumerate(questions):
            print(f"  Question: '{q}'")
            ans = run_rag_with_retry(strat["run_rag"], q)
            answers.append(ans)
            
            ctxs = get_retrieved_contexts(q, strat["name"], chunks=text_chunks)
            contexts_list.append(ctxs)
            
            raw_results.append({
                "Strategy": strat["name"],
                "Question": q,
                "Answer": ans,
                "Ground Truth": ground_truths[idx]
            })
            import time
            time.sleep(4)

        all_datasets[strat["name"]] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        }

    import pandas as pd
    comparison_df = pd.DataFrame(raw_results)
    print("\n--- Generation Comparison ---")
    print(comparison_df.to_string(index=False))

    # Stage 2: Run Ragas Metrics
    print("\n--- Running RAGAS Evaluation Metrics ---")
    all_results = {}
    
    for name, data in all_datasets.items():
        print(f"\nEvaluating {name} with RAGAS...")
        dataset = Dataset.from_dict(data)
        
        from ragas.run_config import RunConfig
        # Strict sequential execution and generous retries for Trial key
        run_config = RunConfig(max_workers=1, max_retries=10, max_wait=30)
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config
            )
            all_results[name] = result.to_pandas()
            print(all_results[name])
        except Exception as e:
            print(f"Warning: RAGAS evaluation failed for {name} (probably rate limit/parsing error): {e}")
            # Fallback mock/empty frame
            all_results[name] = pd.DataFrame(data)

    # Compile a final summary table
    print("\n--- Final RAGAS Summary ---")
    summary_data = []
    for name, df in all_results.items():
        mean_scores = df.mean(numeric_only=True)
        summary_row = {
            "Strategy": name,
            "Faithfulness": mean_scores.get("faithfulness", 0.0),
            "Answer Relevancy": mean_scores.get("answer_relevancy", 0.0),
        }
        summary_data.append(summary_row)
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save to markdown report in local workspace
    markdown_report = "# RAG Evaluation & RAGAS Report\n\n"
    markdown_report += "This report compares the RAG pipeline configurations in the DocBot application.\n\n"
    markdown_report += "## Answer Comparison Table\n\n"
    markdown_report += comparison_df.to_markdown(index=False) + "\n\n"
    markdown_report += "## RAGAS Score Summary Table\n\n"
    markdown_report += summary_df.to_markdown(index=False) + "\n\n"
    
    with open("src/evaluate/evaluate_results.md", "w") as f:
        f.write(markdown_report)
    print("\nSaved evaluation report to src/evaluate/evaluate_results.md")

if __name__ == "__main__":
    main()
