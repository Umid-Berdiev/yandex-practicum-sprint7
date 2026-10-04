# RAG-бот для QuantumForge Software

Проектная работа 7 спринта Яндекс Практикума: бот, который отвечает на вопросы по базе знаний с помощью Retrieval-Augmented Generation.

Бот ищет фрагменты в векторном индексе FAISS, передаёт их локальной LLM и отвечает со ссылками на источники. Если ответа в базе знаний нет, он честно пишет «Я не знаю».

Полный отчёт по всем заданиям — в [Project_template.md](Project_template.md).

## Что внутри

| Компонент | Решение |
|---|---|
| База знаний | 43 статьи вселенной Star Wars с заменёнными названиями (`Darth Vader → Xarn Velgor`, `Death Star → Void Core`), чтобы модель не могла отвечать по памяти |
| Эмбеддинги | `BAAI/bge-m3`, локально, 1024 измерения |
| Векторная БД | FAISS, точный поиск по косинусной близости, 650 чанков |
| LLM | `qwen3:8b` в Ollama; подходит любой OpenAI-совместимый API |
| Промптинг | Few-shot (три примера из базы знаний) и Chain-of-Thought (шаги рассуждения перед ответом) |
| Защита от промпт-инъекций | Pre-prompt, фильтр чанков, очистка текста |
| Интерфейс | Консольный диалог |

## Быстрый старт

Нужны Python 3.12 и [Ollama](https://ollama.com/).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

sh scripts/download_model.sh     # модель эмбеддингов, 2,3 ГБ -> models/bge-m3
ollama serve                     # в отдельном терминале
ollama pull qwen3:8b             # LLM, 5,2 ГБ
cp .env.example .env

python -m rag_bot                # диалог с ботом
```

Индекс уже лежит в репозитории (`faiss_index/`), пересобирать его не нужно.

Один вопрос с показом найденных фрагментов:

```bash
python -m rag_bot --show-context "Кто обучал Corin Vantreil использовать Synth Flux?"
```

## Запуск в Docker

Модель эмбеддингов и Ollama должны быть подготовлены так же, как в быстром старте.

```bash
docker compose build
docker compose run --rm bot                          # диалог
docker compose run --rm bot "Who is Xarn Velgor?"    # один вопрос
```

FAISS — библиотека, а не сервер, поэтому отдельного контейнера у неё нет: индекс подключается томом и загружается в процесс бота. LLM работает в Ollama на хост-машине.

## Примеры

```
Вы: Кто обучал Corin Vantreil использовать Synth Flux?

Рассуждение:
1. Нужно определить, кто обучал Corin Vantreil использовать Synth Flux.
2. В фрагменте [1] говорится, что Corin Vantreil начал обучение в способах Synth Flux
   под руководством Veyari Master Oru-Kei Talvane.
...
Ответ: Corin Vantreil обучался использовать Synth Flux под руководством Veyari Master
Oru-Kei Talvane [1].

Источники:
  [1] knowledge_base/corin_vantreil.md, символы 558–1876
```

```
Вы: Кто такой Дарт Вейдер?

Ответ: Я не знаю. В базе знаний нет информации по этому вопросу.
```

- [screenshots/](screenshots/) — десять скриншотов: пять ответов и пять «Я не знаю».
- [examples/dialogues.md](examples/dialogues.md) — десять диалогов: пять ответов и пять отказов.
- [examples/security_tests.md](examples/security_tests.md) — тесты промпт-инъекции при разных уровнях защиты.

## Структура репозитория

| Путь | Содержимое |
|---|---|
| `rag_bot/` | Бот: настройки, поиск, промпты, цепочка RAG, защита, консольный интерфейс |
| `scripts/download_pages.py` | Скачивание и очистка страниц вики |
| `scripts/replace_terms.py` | Замена терминов, сборка базы знаний и словаря замен |
| `scripts/download_model.sh` | Загрузка модели эмбеддингов |
| `scripts/build_index.py` | Построение индекса FAISS |
| `scripts/search_index.py` | Поиск по индексу без LLM |
| `scripts/run_examples.py` | Прогон десяти демонстрационных вопросов |
| `scripts/run_security_tests.py` | Тесты защиты от промпт-инъекции |
| `knowledge_base/` | Документы базы знаний и вредоносный файл `malicious_note.md` для тестов |
| `terms_map.json` | Словарь замен: исходный термин → вымышленный |
| `faiss_index/` | Готовый индекс и параметры его сборки |
| `examples/` | Логи диалогов и тестов защиты |
| `screenshots/` | Скриншоты работы бота |
| `Dockerfile`, `docker-compose.yml` | Сборка и запуск в Docker |

## Настройки

Задаются в `.env` (образец — [.env.example](.env.example)).

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Адрес OpenAI-совместимого API |
| `LLM_MODEL` | — | Название модели |
| `LLM_API_KEY` | заглушка | Ключ API; для Ollama не нужен |
| `TOP_K` | `5` | Сколько чанков передаётся модели |
| `MIN_SCORE` | `0.5` | Порог близости; ниже него чанк не используется |
| `GUARD_PREPROMPT`, `GUARD_CHUNK_FILTER`, `GUARD_SANITIZE` | `1` | Слои защиты от промпт-инъекций |

## Пересборка базы знаний и индекса

```bash
python scripts/download_pages.py     # страницы вики -> data/raw/
python scripts/replace_terms.py      # замена терминов -> knowledge_base/, terms_map.json
python scripts/build_index.py        # индекс -> faiss_index/
```

`replace_terms.py` пересоздаёт папку `knowledge_base/` целиком, поэтому `malicious_note.md` после него нужно вернуть из git.

## Известные ограничения

- Русский язык у `qwen3:8b` местами неровный; ответ занимает 10–17 секунд.
- Защиту от инъекций обеспечивает фильтр на регулярных выражениях, перефразированную инъекцию он пропустит. Подробности — в разделе «Задание 5» отчёта.
- Docker-образ, собранный за корпоративным прокси с подменой TLS-сертификатов, занимает 9,8 ГБ: вместо CPU-сборки PyTorch ставится полная.

Тексты базы знаний основаны на материалах [Wookieepedia](https://starwars.fandom.com/), лицензия CC BY-SA.
