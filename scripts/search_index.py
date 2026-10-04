"""Search the FAISS index: print the chunks most similar to a query.

Usage: python scripts/search_index.py "Who is Xarn Velgor?" [-k 3]
       python scripts/search_index.py            # runs the built-in example queries
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_bot.index import load_index  # noqa: E402

EXAMPLE_QUERIES = [
    "Who destroyed the Void Core?",
    "What is an arcblade and how is it built?",
    "Кто обучал Корина Вантрейла использовать Synth Flux?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    store = load_index()
    queries = [" ".join(args.query)] if args.query else EXAMPLE_QUERIES
    for query in queries:
        print(f"\n=== Query: {query}")
        for rank, (chunk, score) in enumerate(store.similarity_search_with_score(query, k=args.k), 1):
            meta = chunk.metadata
            preview = " ".join(chunk.page_content.split())[:300]
            print(f"\n[{rank}] score={score:.3f}  {meta['chunk_id']}")
            print(f"    source: {meta['source']}, chars {meta['start_index']}-{meta['end_index']}")
            print(f"    title: {meta['title']} | section: {meta['section'] or '(intro)'}")
            print(f"    {preview}...")


if __name__ == "__main__":
    main()
