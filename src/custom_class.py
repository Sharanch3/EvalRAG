from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi


class MyBM25Retriever(BaseRetriever):
    """Sparse keyword retriever backed by BM25Okapi."""

    docs: List[Document]

    bm25: BM25Okapi | None = None

    k: int = 4

    def model_post_init(self, context):

        tokenized = [doc.page_content.lower().split() for doc in self.docs]

        self.bm25 = BM25Okapi(corpus=tokenized)

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]

        return [self.docs[i] for i in top_k]


class MyEnsembleretriever(BaseRetriever):
    """
    Combines dense + sparse retrievers via
    Reciprocal Rank Fusion (RRF).
    """

    retrievers: List[BaseRetriever]

    weights: List[float]

    k: int = 3

    rrf_k: int = 60

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        all_docs: List[tuple[Document, float]] = []

        for retriever, weight in zip(self.retrievers, self.weights):
            for rank, doc in enumerate(retriever.invoke(query)):
                all_docs.append((doc, weight * (1 / (rank + self.rrf_k))))

        seen: dict[str, tuple[Document, float]] = {}

        for doc, score in all_docs:
            key = doc.page_content

            if key in seen:
                seen[key] = (doc, seen[key][1] + score)

            else:
                seen[key] = (doc, score)

        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[: self.k]]
