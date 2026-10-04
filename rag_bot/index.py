"""Embedding model and FAISS index: shared by the index builder and the bot."""

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from rag_bot import config


def get_embeddings() -> HuggingFaceEmbeddings:
    model = config.EMBEDDING_MODEL_DIR if config.EMBEDDING_MODEL_DIR.exists() else config.EMBEDDING_MODEL
    # Normalized vectors + inner product = cosine similarity.
    return HuggingFaceEmbeddings(
        model_name=str(model),
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def load_index() -> FAISS:
    return FAISS.load_local(
        str(config.INDEX_DIR), get_embeddings(),
        # index.pkl is our own file, produced by scripts/build_index.py.
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


@dataclass
class Chunk:
    """A fragment of the knowledge base found for a query."""

    text: str
    score: float
    chunk_id: str
    source: str
    title: str
    section: str
    start_index: int
    end_index: int

    @property
    def reference(self) -> str:
        section = f", раздел «{self.section}»" if self.section else ""
        return f"{self.source}{section}, символы {self.start_index}–{self.end_index}"


class Retriever:
    """Embeds the query with the index encoder and returns the nearest chunks."""

    def __init__(self, store: FAISS | None = None):
        self.store = store or load_index()

    def search(self, query: str, k: int = config.TOP_K, min_score: float = config.MIN_SCORE) -> list[Chunk]:
        found = self.store.similarity_search_with_score(query, k=k)
        return [
            Chunk(text=doc.page_content, score=float(score),
                  **{key: doc.metadata[key] for key in
                     ("chunk_id", "source", "title", "section", "start_index", "end_index")})
            for doc, score in found if score >= min_score
        ]
