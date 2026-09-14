import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.custom_class import MyBM25Retriever, MyEnsembleretriever

load_dotenv()


VS_DIR = Path("chroma_store")
_vs_instance: Chroma | None = None
_bm25_instance: MyBM25Retriever | None = None


def _get_vector_store() -> Chroma:
    """Create (or return) the singleton Chroma instance."""

    if not VS_DIR.exists():
        VS_DIR.mkdir(parents=True, exist_ok=True)

    global _vs_instance

    if _vs_instance is None:
        ef = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

        _vs_instance = Chroma(collection_name="pdfs", embedding_function=ef, persist_directory="Chroma_store", create_collection_if_not_exists=True)

    return _vs_instance


def _get_bm25() -> MyBM25Retriever:
    """Return the singleton BM25 retriever."""

    global _bm25_instance

    if _bm25_instance is None:
        vs = _get_vector_store()

        data = vs.get(include=["documents", "metadatas"])

        docs = [Document(page_content=text, metadata=metadata or {}) for text, metadata in zip(data["documents"], data["metadatas"])]

        _bm25_instance = MyBM25Retriever(docs=docs, k=4)

    return _bm25_instance


def pdf_loader(file_path: str | Path, chunk_size: int = 600, chunk_overlap: int = 50) -> List[Document]:

    loader = PyMuPDF4LLMLoader(file_path=file_path, mode="page", extract_images=False)

    docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])

    return splitter.split_documents(documents=docs)


def ingest_to_vs(file_path: str) -> None:
    """Parse a PDF and add its chunks to the vector store."""

    global _bm25_instance

    docs = pdf_loader(file_path=file_path)

    vs = _get_vector_store()

    vs.add_documents(documents=docs)

    # Reset to pull fresh data
    _bm25_instance = None


def hybrid_retriever() -> MyEnsembleretriever:
    """
    Return sparce and dense represenataion retrieval. Hybrid search

    """

    vs = _get_vector_store()

    mmr_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.4},
    )

    bm25_retriever = _get_bm25()

    return MyEnsembleretriever(retrievers=[mmr_retriever, bm25_retriever], weights=[0.6, 0.4], k=3)




if __name__ == "__main__":

    # Ingest once
    ingest_to_vs(file_path= Path("data/Openclaw_Research_Report.pdf"))

    # Retriever
    retriever = hybrid_retriever()

    query = "What is moltbot?"
    results = retriever.invoke(query)

    # Display the retrieved documents
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} documents:\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Document {i} ---")
        print("Content:")
        print(doc.page_content)
        print("\n")





