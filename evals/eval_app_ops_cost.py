from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.generator import model, prompt
from src.workflow import RAGWorkflow

load_dotenv()


QUESTIONS = ["What messaging platforms and channels does Moltbot support?", "What LLM providers and models can Moltbot use?", "What did security experts say about the risks of running Moltbot?"]


chain = prompt | model

PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60

REPEATS = 3

USD_TO_INR = 96
QUERIES_PER_DAY = 2000
COST_BUDGET_PER_QUERY_USD = 0.0015


def measure_tokens(pipeline: RAGWorkflow, question: str) -> dict:

    docs: List[Document] = pipeline.retriever.invoke(question)
    context_text: str = "\n".join(doc.page_content for doc in docs)

    ai_msg: AIMessage = chain.invoke({"question": question, "context": context_text})

    usage = ai_msg.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    details = usage.get("input_token_details", {})
    cached_tokens = details.get("cache_read", 0)

    return {"input": input_tokens, "output": output_tokens, "cached": cached_tokens}


def cost_usd(input_tokens: int, output_tokens: int, cached_tokens: int) -> dict:

    cache_miss = max(input_tokens - cached_tokens, 0)
    input_cost = (cache_miss / 1_000_000) * PRICE_INPUT_PER_1M

    cached_cost = (cached_tokens / 1_000_000) * PRICE_CACHED_INPUT_PER_1M

    output_cost = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M

    return {"input": input_cost, "cached": cached_cost, "output": output_cost, "total": (input_cost + cached_cost + output_cost)}


def benchmark(pipeline: RAGWorkflow) -> List[dict]:

    rows = []
    print("Measuring token usage...")
    for question in QUESTIONS:
        for _ in range(REPEATS):
            tokens_info: dict = measure_tokens(pipeline=pipeline, question=question)
            cost: dict = cost_usd(input_tokens=tokens_info["input"], output_tokens=tokens_info["output"], cached_tokens=tokens_info["cached"])

            rows.append({**tokens_info, **{f"cost_{k}": v for k, v in cost.items()}})

    return rows


def _average(rows: List[dict], key: str) -> float:

    return sum(r[key] for r in rows) / len(rows)


def report(rows: List[dict]) -> None:

    n = len(rows)
    avg_input = _average(rows=rows, key="input")
    avg_output = _average(rows=rows, key="output")
    avg_cached = _average(rows=rows, key="cached")
    avg_cost = _average(rows=rows, key="cost_total")

    min_cost = min(r["cost_total"] for r in rows)
    max_cost = max(r["cost_total"] for r in rows)

    # avg_cost_input = _average(rows=rows, key="cost_input") + _average(rows=rows, key="cost_cached")
    avg_cost_output = _average(rows=rows, key="cost_output")
    output_share = 100 * avg_cost_output / (avg_cost if avg_cost else 0)

    print("\n" + "=" * 70)
    print(f"COST  (gpt-4o-mini @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)")
    print("=" * 70)
    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_input:8.0f}   ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_output:8.0f}")

    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   <- tight range = cost is stable, unlike latency")
    print(f"   input vs output     : {100 - output_share:.0f}% input / {output_share:.0f}% output (output is 4x the rate -> long answers dominate)")
    print("-" * 70)

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  ${avg_cost:.6f}   [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push the")
    print("real bill BELOW this estimate -- watch the 'cached' count grow online.")


def main() -> None:
    pipeline = RAGWorkflow()
    rows = benchmark(pipeline=pipeline)
    report(rows=rows)


if __name__ == "__main__":
    main()
