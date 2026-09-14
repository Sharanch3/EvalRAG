import json
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.generator import generator

GOLDENS_PATH = Path("goldens/generator_goldens.json")
JUDGE_MODEL = "gpt-4o-mini"


def eval_generator() -> None:
    if not GOLDENS_PATH.exists():
        raise FileNotFoundError(f"Goldens file not foud at {GOLDENS_PATH}")
    with open(GOLDENS_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    test_cases = []

    for g in goldens:
        context = g["ideal_context"]
        answer = generator(question=g["input"], context=context)

        test_cases.append(
            LLMTestCase(input=g["input"], actual_output=answer, retrieval_context=context)
        )

    metrics = [
        FaithfulnessMetric(
            threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True
        ),
        AnswerRelevancyMetric(
            threshold=0.7, model=JUDGE_MODEL, include_reason=True, async_mode=True
        ),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=5),
    )

    return None


if __name__ == "__main__":
    eval_generator()
