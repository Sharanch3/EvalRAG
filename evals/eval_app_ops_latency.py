import math
import time
from typing import List

import numpy as np

from src.generator import generator, stream_generator
from src.workflow import RAGWorkflow

QUESTIONS = ["What messaging platforms and channels does Moltbot support?", "What LLM providers and models can Moltbot use?", "What did security experts say about the risks of running Moltbot?"]

REPEATS = 5
WARMUP_RUNS = 2

MEASURE_TTFT = True
STAGE_LEVEL = True

SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200


def run_end_to_end(pipeline: RAGWorkflow, query: str) -> str:

    results: dict = pipeline.invoke(query=query)

    return results["answer"]


def run_stages(pipeline: RAGWorkflow, query: str) -> tuple[str, dict]:

    t0 = time.perf_counter()
    docs = pipeline.retriever.invoke(query)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()

    answer = generator(question=query, context=context)
    t2 = time.perf_counter()

    return answer, {"retrieval": round((t1 - t0) * 1000, 3), "generator": round((t2 - t1) * 1000, 3)}


def run_stages_streaming(pipeline: RAGWorkflow, query: str) -> tuple[str, dict]:

    t0 = time.perf_counter()
    docs = pipeline.retriever.invoke(query)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()

    first_token_time = None
    tokens = []
    for token in stream_generator(question=query, context=context):
        if first_token_time is None:
            first_token_time = time.perf_counter()

        tokens.append(token)

    t2 = time.perf_counter()

    answer = "".join(tokens)
    ttft_ms = round((first_token_time - t1) * 1000, 3) if first_token_time else float("nan")

    return answer, {"retrieval": round((t1 - t0) * 1000, 3), "generator": round((t2 - t1) * 1000, 3), "ttft_ms": ttft_ms}


def percentile(values: List[float], p: int) -> float:
    only_values: List[float] = [v for v in values if not math.isnan(v)]

    if not only_values:
        return float("nan")

    return np.percentile(only_values, p)


def benchmark(pipeline: RAGWorkflow) -> dict:

    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...")
    for i in range(WARMUP_RUNS):
        run_end_to_end(pipeline=pipeline, query=QUESTIONS[i])

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []
    print("Measuring...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            start = time.perf_counter()

            if MEASURE_TTFT:
                answer, stage = run_stages_streaming(pipeline=pipeline, query=question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generator"])
                ttft_ms.append(stage["ttft_ms"])

            elif STAGE_LEVEL:
                answer, stage = run_stages(pipeline=pipeline, query=question)
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generator"])

            else:
                answer = run_end_to_end(pipeline=pipeline, query=question)

            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)

            total_ms.append(elapsed_ms)
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


def statistics(samples: List[float]) -> dict:

    only_numbers: List[float] = [s for s in samples if not math.isnan(s)]

    return {
        "n": len(only_numbers),
        "mean": sum(only_numbers) / len(only_numbers),
        "p50": percentile(values=only_numbers, p=50),
        "p95": percentile(values=only_numbers, p=95),
        "p99": percentile(values=only_numbers, p=99),
        "min": min(only_numbers),
        "max": max(only_numbers),
    }


def pretty_print(label: str, stats: dict) -> None:

    print(f"{label:<12} | n={stats['n']:<3}mean={stats['mean']:7.1f}  p50={stats['p50']:7.1f}  p95={stats['p95']:7.1f}  p99={stats['p99']:7.1f}  min={stats['min']:7.1f}  max={stats['max']:7.1f}")


def slo_line(label: str, p95: float, budget: int) -> None:

    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO: {label:<22} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms   [{verdict}]")


def report(results: dict) -> None:
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)
    print(f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} {'p95':>11} {'p99':>11} {'min':>11} {'max':>11}")
    print("-" * 78)

    total: dict = statistics(samples=results["total"])
    pretty_print(label="end-to-end", stats=total)
    if results["ttft"]:
        pretty_print(label="ttft", stats=statistics(results["ttft"]))
    if results["retrieval"]:
        pretty_print(label="retrieval", stats=statistics(results["retrieval"]))
        pretty_print(label="generation", stats=statistics(results["generation"]))
    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars (latency scales with output length -- keep in mind when comparing configs)")

    print("=" * 78)
    slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        slo_line(label="first token (perceived)", p95=statistics(results["ttft"])["p95"], budget=SLO_TTFT_P95_MS)
    print("=" * 78)


def main() -> None:
    pipeline = RAGWorkflow()
    results = benchmark(pipeline=pipeline)
    report(results=results)


if __name__ == "__main__":
    main()
