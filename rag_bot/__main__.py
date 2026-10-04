"""Console interface of the bot.

Usage: python -m rag_bot                       # interactive dialogue
       python -m rag_bot "Who is Xarn Velgor?"   # one question
       python -m rag_bot --show-context ...      # also print the retrieved chunks
"""

import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from rag_bot import config  # noqa: E402
from rag_bot.pipeline import Answer, RagBot  # noqa: E402


def print_answer(result: Answer, show_context: bool) -> None:
    if show_context:
        print("\nНайденные фрагменты:")
        if not result.chunks:
            print(f"  нет фрагментов с близостью не ниже {config.MIN_SCORE}")
        for number, chunk in enumerate(result.chunks, 1):
            preview = " ".join(chunk.text.split())[:160]
            print(f"  [{number}] {chunk.score:.3f}  {chunk.chunk_id}: {preview}...")
    if result.reasoning:
        print(f"\nРассуждение:\n{result.reasoning}")
    print(f"\nОтвет: {result.answer}")
    if result.sources:
        print("\nИсточники:")
        for chunk in result.sources:
            number = result.chunks.index(chunk) + 1
            print(f"  [{number}] {chunk.reference}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-бот по базе знаний")
    parser.add_argument("question", nargs="*", help="вопрос; без него запускается диалог")
    parser.add_argument("--show-context", action="store_true", help="показать найденные фрагменты")
    args = parser.parse_args()

    bot = RagBot()
    if args.question:
        print_answer(bot.ask(" ".join(args.question)), args.show_context)
        return

    print(f"RAG-бот. Модель: {config.LLM_MODEL}. Задайте вопрос или введите «выход».")
    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"выход", "exit", "quit"}:
            break
        if question:
            print_answer(bot.ask(question), args.show_context)


if __name__ == "__main__":
    main()
