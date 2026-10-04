"""Build a FAISS vector index from the knowledge base.

Input:  knowledge_base/*.md
Output: faiss_index/index.faiss  - vectors
        faiss_index/index.pkl    - chunk texts and metadata
        faiss_index/index_meta.json - model, sizes and timings of this build

Only the embedding model is used here, no LLM.

Usage: python scripts/build_index.py
"""

import json
import os
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "faiss_index"

EMBEDDING_MODEL = "BAAI/bge-m3"
# Local copy of the model (see README: downloaded once with curl). If the folder is
# missing, the model is downloaded from the Hugging Face Hub by name.
MODEL_DIR = Path(os.environ.get("EMBEDDING_MODEL_DIR", ROOT / "models" / "bge-m3"))
# ~1500 characters is about 250 words: within the 100-300 words range.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
# Split on section headings first, then paragraphs, sentences and words.
SEPARATORS = ["\n## ", "\n### ", "\n\n", ". ", " ", ""]


def get_embeddings() -> HuggingFaceEmbeddings:
    # Normalized vectors + inner product = cosine similarity.
    return HuggingFaceEmbeddings(
        model_name=str(MODEL_DIR) if MODEL_DIR.exists() else EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def section_at(text: str, position: int) -> str:
    """Heading of the section that contains the given position ('' for the intro)."""
    section = ""
    for match in re.finditer(r"^#{2,3} (.+)$", text, flags=re.MULTILINE):
        if match.start() > position:
            break
        section = match.group(1)
    return section


def load_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS, add_start_index=True,
    )
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        document = Document(page_content=text, metadata={"source": f"{KB_DIR.name}/{path.name}", "title": title})
        for number, chunk in enumerate(splitter.split_documents([document])):
            start = chunk.metadata["start_index"]
            chunk.metadata.update({
                "chunk_id": f"{path.stem}#{number}",
                "chunk_number": number,
                # A chunk may start with the newline that precedes its heading.
                "section": section_at(text, start + 2),
                "end_index": start + len(chunk.page_content),
            })
            chunk.page_content = chunk.page_content.strip()
            chunks.append(chunk)
    return chunks


def main() -> None:
    started = time.perf_counter()
    chunks = load_chunks()
    documents = len({c.metadata["source"] for c in chunks})
    print(f"documents: {documents}, chunks: {len(chunks)}")

    embeddings = get_embeddings()
    model_loaded = time.perf_counter()

    # The title is embedded together with the text: a chunk from the middle of an
    # article often does not mention the entity it describes.
    texts = [f"{c.metadata['title']}\n{c.page_content}" for c in chunks]
    vectors = embeddings.embed_documents(texts)
    embedded = time.perf_counter()

    # The stored text is the chunk itself; the title lives in metadata.
    store = FAISS.from_embeddings(
        text_embeddings=[(c.page_content, v) for c, v in zip(chunks, vectors)],
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
        ids=[c.metadata["chunk_id"] for c in chunks],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    store.save_local(str(INDEX_DIR))
    finished = time.perf_counter()

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": store.index.d,
        "index_type": type(store.index).__name__,
        "similarity": "cosine (normalized vectors, inner product)",
        "documents": documents,
        "chunks": store.index.ntotal,
        "chunk_size_chars": CHUNK_SIZE,
        "chunk_overlap_chars": CHUNK_OVERLAP,
        "avg_chunk_words": round(sum(len(c.page_content.split()) for c in chunks) / len(chunks)),
        "model_load_seconds": round(model_loaded - started, 1),
        "embedding_seconds": round(embedded - model_loaded, 1),
        "total_seconds": round(finished - started, 1),
    }
    (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
