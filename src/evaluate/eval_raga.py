import warnings
warnings.filterwarnings('ignore')
from langchain.embeddings import CohereEmbeddings
from langchain.chat_models import ChatCohere
from langchain.vectorstores.chroma import Chroma
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# todo
# Loading the sample data using the TextLoader
# split the pages into chunks
# Define LLM
# Define prompt template
# traversing each question and passing into the chain to get answer from the system
# Preparing the dataset

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
print(df)
