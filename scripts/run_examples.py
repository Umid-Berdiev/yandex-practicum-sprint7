"""Run the bot on a fixed list of questions and save the dialogues to examples/dialogues.md.

Usage: python scripts/run_examples.py
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_bot import config  # noqa: E402
from rag_bot.pipeline import RagBot  # noqa: E402

QUESTIONS = [
    # the answer is in the knowledge base
    "Кто обучал Corin Vantreil использовать Synth Flux?",
    "Что такое Void Core и как он был уничтожен?",
    "Who is Xarn Velgor and who was he before?",
    "Из каких компонентов состоит arcblade?",
    "На каком корабле летал Dax Varro и кто был его вторым пилотом?",
    # the knowledge base has no answer
    "Какой любимый напиток у Olvek?",
    "Сколько стоит билет на транспорт от Sarrakesh до Velaris?",
    "Кто такой Дарт Вейдер?",
    "Как сбросить пароль в Jira?",
]


def main() -> None:
    bot = RagBot()
    lines = ["# Примеры диалогов с ботом", "",
             f"Модель: `{config.LLM_MODEL}`, эмбеддинги: `{config.EMBEDDING_MODEL}`, "
             f"фрагментов на запрос: {config.TOP_K}, порог близости: {config.MIN_SCORE}.", ""]
    for number, question in enumerate(QUESTIONS, 1):
        started = time.perf_counter()
        result = bot.ask(question)
        seconds = time.perf_counter() - started
        best = f"{result.chunks[0].score:.3f}" if result.chunks else f"ниже {config.MIN_SCORE}"
        lines += [f"## {number}. {question}", "",
                  f"Найдено фрагментов: {len(result.chunks)}, лучшая близость: {best}, "
                  f"время ответа: {seconds:.1f} с, бот ответил «Я не знаю»: {'да' if result.no_answer else 'нет'}.", "",
                  "**Рассуждение:**", "", result.reasoning, "",
                  f"**Ответ:** {result.answer}", ""]
        if result.sources:
            lines += ["**Источники:**", ""]
            lines += [f"- [{result.chunks.index(c) + 1}] {c.reference}" for c in result.sources]
            lines.append("")
        print(f"{number}. {question}\n   -> {result.answer}\n")
    path = ROOT / "examples" / "dialogues.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
