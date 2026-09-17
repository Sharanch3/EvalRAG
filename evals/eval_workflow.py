import json
from pathlib import Path
from typing import List

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.workflow import RAGWorkflow

GOLDENS_PATH = Path("goldens/workflow_goldens.json")
JUDGE_MODEL = "gpt-4o-mini"


def eval_workflow() -> None:
    if not GOLDENS_PATH.exists():
        raise FileNotFoundError(f"Golden file not found: {GOLDENS_PATH}")
    else:
        with open(GOLDENS_PATH, "r", encoding="utf-8") as f:
            goldens = json.load(f)

    pipeline = RAGWorkflow()

    test_cases: List[LLMTestCase] = []

    for g in goldens:
        results = pipeline.invoke(query=g["input"])

        test_cases.append(
            LLMTestCase(
                input=g["input"],
                actual_output=results["answer"],
                retrieval_context=results["context"],
            )
        )

    metrics = [
        ContextualRelevancyMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True),
        FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True),
        AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=True, max_concurrent=5),
    )

    return None


if __name__ == "__main__":
    eval_workflow()
