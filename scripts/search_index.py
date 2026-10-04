"""Search the FAISS index: print the chunks most similar to a query.

Usage: python scripts/search_index.py "Who is Xarn Velgor?" [-k 3]
       python scripts/search_index.py            # runs the built-in example queries
"""

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from build_index import INDEX_DIR, get_embeddings

EXAMPLE_QUERIES = [
    "Who destroyed the Void Core?",
    "What is an arcblade and how is it built?",
    "Кто обучал Корина Вантрейла использовать Synth Flux?",
]


def load_index() -> FAISS:
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(),
        # index.pkl is our own file, produced by build_index.py.
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


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
