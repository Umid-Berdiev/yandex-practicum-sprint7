"""Protection against prompt injections that arrive inside knowledge base documents.

Three independent layers, each can be switched off with an environment variable:
1. GUARD_PREPROMPT    - system message: never follow commands found inside documents.
2. GUARD_CHUNK_FILTER - retrieved chunks with potentially malicious content are dropped.
3. GUARD_SANITIZE     - instruction-like constructs are cut out of the chunk text.
"""

import os
import re
from dataclasses import dataclass

# Phrases typical for an attempt to give orders to the model or to make it reveal secrets.
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous\s+|prior\s+|above\s+)?(?:instructions?|rules?|prompts?)",
    r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous\s+|prior\s+|above\s+)?(?:instructions?|rules?)",
    r"forget\s+(?:all\s+|everything|your\s+)(?:\s*previous)?\s*(?:instructions?|rules?)?",
    r"you\s+are\s+now\s+",
    r"system\s+prompt",
    r"(?:^|[.!?\n]\s*)(?:output|print|say|respond\s+with|reply\s+with)\s*:",
    r"игнорируй\s+(?:все\s+|любые\s+)?(?:предыдущие\s+)?(?:инструкции|правила|указания)",
    r"забудь\s+(?:все\s+)?(?:предыдущие\s+)?(?:инструкции|правила|указания)",
    r"(?:^|[.!?\n]\s*)(?:выведи|напечатай|ответь)\s*:",
    r"системн\w+\s+промпт",
]
_PATTERN = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), flags=re.IGNORECASE)

# Credentials written in plain text: "password: ...", "пароль root: ...", "api_key=...".
# Such a chunk is dangerous even without any injection phrase around it.
SECRET_PATTERN = re.compile(
    r"(?:\w*парол\w*|password|passwd|api[_ -]?key|secret[_ -]?key|access[_ -]?token|токен\w*)"
    r"(?:\s+[\w-]+){0,3}\s*[:=]\s*\S+",
    flags=re.IGNORECASE,
)


def _flag(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Guards:
    preprompt: bool = True
    chunk_filter: bool = True
    sanitize: bool = True

    @classmethod
    def from_env(cls) -> "Guards":
        return cls(preprompt=_flag("GUARD_PREPROMPT"), chunk_filter=_flag("GUARD_CHUNK_FILTER"),
                   sanitize=_flag("GUARD_SANITIZE"))

    def describe(self) -> str:
        names = [title for enabled, title in ((self.preprompt, "pre-prompt"),
                                               (self.chunk_filter, "фильтр чанков"),
                                               (self.sanitize, "очистка текста")) if enabled]
        return ", ".join(names) or "без защиты"


def is_suspicious(text: str) -> bool:
    """Post-retrieval check: does the chunk look like an instruction for the model or hold a secret?"""
    return _PATTERN.search(text) is not None or SECRET_PATTERN.search(text) is not None


def sanitize(text: str) -> str:
    """Cut instruction-like constructs such as "Ignore all instructions" out of the text."""
    return _PATTERN.sub(" [удалено] ", text).strip()
