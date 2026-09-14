import json
from pathlib import Path
from typing import List

from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.synthesizer.synthesizer import Synthesizer

DOC_PATH = Path("data/Openclaw_Research_Report.pdf").as_posix()
FILE_PATH = Path("goldens/retriever_goldens.json").as_posix()


def generate_goldens(documents_path: List[str]) -> None:

    syntheseizer = Synthesizer()

    context_config = ContextConstructionConfig(
        embedder="text-embedding-3-small",
        critic_model="gpt-4o-mini",
        max_contexts_per_document=6,
        min_contexts_per_document=2,
        chunk_size=1024,
        chunk_overlap=100,
    )

    goldens = syntheseizer.generate_goldens_from_docs(
        document_paths=documents_path,
        include_expected_output=True,
        max_goldens_per_context=2,
        context_construction_config=context_config,
    )

    pairs = [
        {"id": f"g{i:03d}", "input": g.input, "expected_output": g.expected_output}
        for i, g in enumerate(goldens, start=1)
        if g.input and g.expected_output
    ]

    with open(file=FILE_PATH, mode="w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    return None


def main():
    generate_goldens(documents_path=[DOC_PATH])


if __name__ == "__main__":
    main()
