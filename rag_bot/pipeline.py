"""RAG pipeline: question -> retrieval -> prompt -> LLM -> answer with sources."""

import re
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from rag_bot import config
from rag_bot.index import Chunk, Retriever
from rag_bot.prompts import NO_ANSWER, NO_ANSWER_EN, build_messages


@dataclass
class Answer:
    question: str
    reasoning: str
    answer: str
    # True when the bot honestly says it does not know.
    no_answer: bool
    # Chunks given to the LLM as context.
    chunks: list[Chunk] = field(default_factory=list)
    # Chunks the answer refers to.
    sources: list[Chunk] = field(default_factory=list)


def get_llm() -> ChatOpenAI:
    if not config.LLM_MODEL:
        raise SystemExit("LLM_MODEL is not set: copy .env.example to .env and fill it in")
    return ChatOpenAI(
        model=config.LLM_MODEL, base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
        temperature=config.LLM_TEMPERATURE, timeout=120,
        # The prompt already asks for explicit reasoning steps, so the hidden "thinking"
        # of reasoning models is switched off by default: it only slows the answer down.
        reasoning_effort=config.LLM_REASONING_EFFORT or None,
    )


class RagBot:
    def __init__(self, retriever: Retriever | None = None, llm: ChatOpenAI | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm()

    def ask(self, question: str) -> Answer:
        # 1. The query is embedded with the same encoder as the index; nearest chunks are returned.
        chunks = self.retriever.search(question)

        # 2. Nothing similar enough in the knowledge base: answer without calling the LLM,
        #    so that the model has no chance to answer from its own memory.
        if not chunks:
            return Answer(
                question=question, no_answer=True,
                reasoning="В базе знаний не нашлось фрагментов, достаточно близких к вопросу.",
                answer=f"{NO_ANSWER}. В базе знаний нет информации по этому вопросу.",
            )

        # 3. Prompt = system instructions (CoT) + few-shot examples + context + question.
        messages = build_messages(question, chunks)

        # 4. Generation.
        text = self.llm.invoke(messages).content
        reasoning, answer = split_response(text)

        # 5. Sources: chunks cited as [n] in the final answer (the reasoning also
        #    mentions chunks that turned out to be irrelevant).
        no_answer = answer.lower().replace("’", "'").startswith((NO_ANSWER.lower(), NO_ANSWER_EN.lower()))
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [] if no_answer else [c for n, c in enumerate(chunks, 1) if n in cited]
        return Answer(question=question, reasoning=reasoning, answer=answer,
                      no_answer=no_answer, chunks=chunks, sources=sources)


def split_response(text: str) -> tuple[str, str]:
    """Split the model output into the reasoning steps and the final answer."""
    # Local reasoning models may wrap hidden thoughts in <think> tags.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Headings are "Рассуждение:/Ответ:" in Russian answers and "Reasoning:/Answer:" in English ones.
    parts = re.split(r"^\**(?:Ответ|Answer)\**:\**", text, flags=re.MULTILINE)
    if len(parts) < 2:
        return "", text
    reasoning = re.sub(r"^\s*\**(?:Рассуждение|Reasoning)\**:\**\s*", "", parts[0]).strip()
    return reasoning, parts[-1].strip()
