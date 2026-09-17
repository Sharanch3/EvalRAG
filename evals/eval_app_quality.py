import json
from pathlib import Path
from typing import List

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.workflow import RAGWorkflow

GOLDENS_PATH = Path("goldens/app_quality_goldens.json")
JUDGE_MODEL = "gpt-4o-mini"

pipeline = RAGWorkflow(k=3)


def eval_app_quality() -> None:

    if not GOLDENS_PATH.exists():
        raise FileNotFoundError(f"Goldens file not foud at {GOLDENS_PATH}")

    with open(GOLDENS_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    test_cases: List[LLMTestCase] = []

    for g in goldens:
        results = pipeline.invoke(query=g["input"])

        test_cases.append(LLMTestCase(input=g["input"], expected_output=g["expected_output"], actual_output=results["answer"]))

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=("All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=("Mostly correct but contains one minor inaccuracy."),
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=("Contains a clear factual error or a claim that contradicts the expected output."),
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=("Addresses essentially all key points in the expected output."),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome=("Covers the main key points but misses one or more."),
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=("Misses several key points and only partially covers the expected output."),
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the writing style, clarity, tone, and presentation of the actual output.",
            "Do not judge factual correctness, retrieval quality, completeness, or faithfulness to the source.",
            "Reward answers that explain research-paper concepts clearly and accessibly while maintaining a professional and technically appropriate tone.",
            "Reward answers that translate dense academic language into understandable explanations without unnecessarily oversimplifying important technical concepts.",
            "Reward a logical explanatory flow where the main idea is introduced clearly before diving into technical terminology, methodology, equations, or detailed findings.",
            "Reward concise unpacking of technical terms when they are necessary for understanding the answer.",
            "Reward a response style that feels like a knowledgeable researcher or technical expert explaining a paper to an informed reader, rather than simply copying academic language from the paper.",
            "Allow appropriate structure such as short paragraphs, headings, or bullet points when they improve readability.",
            "Do not penalize structured formatting by itself.",
            "Penalize answers that blindly reproduce dense academic jargon, sound excessively formal or robotic, or present technical information without explanation.",
            "Penalize answers that are difficult to follow because they jump between ideas without a clear narrative or explanatory flow.",
            "Do NOT reward or penalize based on factual correctness, retrieval relevance, completeness, citation accuracy, or answer length.",
            "Judge only style, clarity, tone, and presentation.",
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Excellent research-explanation style. Clear, "
                    "professional, and easy to follow. Complex ideas "
                    "are translated into accessible language while "
                    "preserving technical meaning. The response has "
                    "a strong logical flow and explains concepts rather "
                    "than merely repeating academic wording."
                ),
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "Good and professional explanatory style. The answer is clear and readable, with mostly effective handling of technical concepts. Minor jargon or slightly dense sections may exist."
                ),
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome=("Understandable but somewhat dense, academic, mechanical, or inconsistently structured. Technical concepts may occasionally be insufficiently explained."),
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome=("Poor presentation style. The response is difficult to follow, excessively jargon-heavy, robotic, or largely copies dense academic language without explaining it."),
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[
            correctness,
            completeness,
            style,
        ],
        async_config=AsyncConfig(run_async=True, max_concurrent=5),
    )


if __name__ == "__main__":
    eval_app_quality()
