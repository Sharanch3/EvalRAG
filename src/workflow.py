from typing import List

from src.generator import generator
from src.retriever import get_retriever


class RAGWorkflow:
    def __init__(self, k: int = 3):
        self.retriever = get_retriever(k=k)

    def invoke(self, query: str):

        docs = self.retriever.invoke(query)

        context: List[str] = [doc.page_content.strip() for doc in docs]

        answer: str = generator(question=query, context=context)

        return {"query": query, "context": context, "answer": answer}


if __name__ == "__main__":
    pipeline = RAGWorkflow(k=3)

    results = pipeline.invoke(query="What is Moltbot?")

    print(f"QUERY: {results['query']}\n")
    print(f"ANSWER: {results['answer']}\n\n")
    print("CONTEXT CHUNKS:")
    for i, chunk in enumerate(results["context"], start= 1):
        print(f"[CHUNK-{i}] {chunk[:120]}...")
