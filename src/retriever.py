from src.custom_class import MyEnsembleretriever
from src.vector_store import hybrid_retriever


def get_retriever() -> MyEnsembleretriever:
    """Return the Ensemble retriever backed by Chroma."""

    retriever = hybrid_retriever()

    return retriever


if __name__ == "__main__":
    # Retriever
    retriever = get_retriever()

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
