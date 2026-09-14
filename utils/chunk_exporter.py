import json
from pathlib import Path
from typing import List

from src.vector_store import _get_vector_store


def get_chunks():
    vs = _get_vector_store()

    results = vs.get(include=["documents"])

    chunks: List[dict] = [
        {"id": f"g{i:03d}", "goldens_chunks": chunk}
        for i, chunk in enumerate(results["documents"], start=1)
    ]

    return chunks


def export_chunks():
    all_chunks = get_chunks()

    output_path = Path("utils/exported_chunks.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(all_chunks)} chunks to {output_path}")


if __name__ == "__main__":
    export_chunks()
