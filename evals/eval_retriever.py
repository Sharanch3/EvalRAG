import json
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from src.retriever import get_retriever

load_dotenv()


GOLDENS_PATH = Path("goldens/retriever_goldens.json")
JUDGE_MODEL = "gpt-4o-mini"


def eval_retriever() -> None:
    if not GOLDENS_PATH.exists():
        raise FileNotFoundError(f"Goldens file not found at {GOLDENS_PATH}")
    with open(GOLDENS_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    retriever = get_retriever()

    test_cases = []

    for g in goldens:
        retrieved = retriever.invoke(g["input"])

        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["input"],
                expected_output=g["expected_output"],
                retrieval_context=retrieval_context,
                actual_output=None,
            )
        )

    metrics = [
        ContextualRecallMetric(
            threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True
        ),
        ContextualPrecisionMetric(
            threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True
        ),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=5),
        hyperparameters={
            "retriever": "hybrid",
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 550,
            "chunk_overlap": 20,
            "top_k": 3,
            "judge_model": "gpt-4o-mini",
            "golden_set": GOLDENS_PATH.as_posix(),
        },
    )

    return None


if __name__ == "__main__":
    eval_retriever()
