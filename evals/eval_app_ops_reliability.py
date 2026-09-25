import time
from typing import Callable

from dotenv import load_dotenv

from src.workflow import RAGWorkflow

load_dotenv()


QUESTIONS = ["What messaging platforms and channels does Moltbot support?", "What LLM providers and models can Moltbot use?", "What did security experts say about the risks of running Moltbot?"]

REPEATS = 5
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


class Reliability:
    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0


def call_with_retries(func: Callable, reliability: Reliability):

    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = func()
            reliability.successes += 1

            return result

        except Exception as e:
            if attempt < MAX_RETRIES:
                reliability.retries += 1

                time.sleep(BACKOFF_BASE_S * (2**attempt))

            else:
                reliability.failures += 1
                print(f"FAILED after {MAX_RETRIES} retries: {e}")

                return None


def benchmark(pipeline: RAGWorkflow):

    reliability = Reliability()
    print("Measuring reliability...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            call_with_retries(func=lambda: pipeline.invoke(question), reliability=reliability)

    return reliability


def report(rel: Reliability) -> None:

    success_rate = 100 * rel.successes / rel.calls if rel.calls else 0

    error_rate = 100 * rel.failures / rel.calls if rel.calls else 0

    retry_rate = 100 * rel.retries / rel.calls if rel.calls else 0

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)

    print(f"total requests : {rel.calls}")
    print(f"successful     : {rel.successes}")
    print(f"failed         : {rel.failures}")

    print("-" * 60)

    print(f"success rate   : {success_rate:.2f}%")
    print(f"error rate     : {error_rate:.2f}%")
    print(f"retry rate     : {retry_rate:.2f}%")

    print("=" * 60)


def main():

    pipeline = RAGWorkflow()

    reliability = benchmark(pipeline=pipeline)

    report(reliability)


if __name__ == "__main__":
    main()
