from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from pydantic import Field

from .schemas import StrictModel


class InvariantSettings(StrictModel):
    emoji_enabled: bool = True
    min_emojis: int = Field(default=3, ge=1, le=10)
    uppercase_enabled: bool = True
    sentence_limit_enabled: bool = True
    max_sentences: int = Field(default=3, ge=1, le=10)


class InvariantSnapshot(InvariantSettings):
    revision: int = Field(ge=1)


class InvariantUpdateRequest(InvariantSettings):
    revision: int = Field(ge=1)


class InvariantSettingsConflict(ValueError):
    pass


class InvariantViolation(ValueError):
    def __init__(self, rules: tuple[str, ...], settings: InvariantSettings) -> None:
        self.rules = rules
        self.settings = settings
        super().__init__(", ".join(rules))

    def refusal(self) -> str:
        text = (
            "Не могу выполнить запрос в таком формате: действуют правила «"
            + "», «".join(self.rules)
            + "»; изменить их можно в меню «Инварианты»"
        )
        # One sentence, even when the configurable limit is one.
        return InvariantPolicy(self.settings).format(text)


class SQLiteInvariantRepository:
    """Product settings independent of profiles, memory and chat history."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS product_invariants (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        revision INTEGER NOT NULL,
                        settings_json TEXT NOT NULL
                    )""",
                )
                connection.execute(
                    "INSERT OR IGNORE INTO product_invariants VALUES (1, 1, ?)",
                    (InvariantSettings().model_dump_json(),),
                )
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path, timeout=10)

    def get(self) -> InvariantSnapshot:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT revision, settings_json FROM product_invariants WHERE id = 1",
            ).fetchone()
            return InvariantSnapshot(**json.loads(row[1]), revision=row[0])
        finally:
            connection.close()

    def update(self, request: InvariantUpdateRequest) -> InvariantSnapshot:
        settings = request.model_dump(exclude={"revision"})
        connection = self._connect()
        try:
            with connection:
                changed = connection.execute(
                    """UPDATE product_invariants SET settings_json = ?, revision = revision + 1
                    WHERE id = 1 AND revision = ?""",
                    (json.dumps(settings, ensure_ascii=False), request.revision),
                ).rowcount
                if not changed:
                    raise InvariantSettingsConflict(
                        "Правила уже изменены. Откройте меню заново и повторите изменение",
                    )
            return InvariantSnapshot(**settings, revision=request.revision + 1)
        finally:
            connection.close()


# Count visible emoji sequences, not variation selectors, skin tones or ZWJ parts.
_EMOJI_BASE = (
    r"(?![\U0001F3FB-\U0001F3FF\U0001F1E6-\U0001F1FF])"
    r"[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF]"
)
_EMOJI = re.compile(
    r"[0-9#*]\ufe0f?\u20e3|[\U0001F1E6-\U0001F1FF]{2}|"
    + _EMOJI_BASE + r"[\ufe0e\ufe0f]?[\U0001F3FB-\U0001F3FF]?"
    + r"(?:\u200d" + _EMOJI_BASE + r"[\ufe0e\ufe0f]?[\U0001F3FB-\U0001F3FF]?)*",
)


def emoji_count(text: str) -> int:
    return len(_EMOJI.findall(text))


def sentence_count(text: str) -> int:
    # Decimal points and list numbering are not sentence boundaries. Emoji-only
    # tails do not become an extra sentence after the final punctuation.
    text = re.sub(r"(?<=\d)\.(?=\d)|(?m:^\s*\d+\.(?=\s))", "", text)
    return sum(
        any(char.isalnum() for char in part)
        for part in re.split(r"[.!?…]+", text)
    )


class InvariantPolicy:
    """Deterministic enforcement of the three editable product rules."""

    def __init__(self, settings: InvariantSettings | None) -> None:
        self.settings = settings

    def descriptions(self) -> tuple[str, ...]:
        settings = self.settings
        if settings is None:
            return ()
        rules = []
        if settings.emoji_enabled:
            rules.append(f"Минимум {settings.min_emojis} эмодзи")
        if settings.uppercase_enabled:
            rules.append("Только ВЕРХНИЙ РЕГИСТР")
        if settings.sentence_limit_enabled:
            rules.append(f"Не больше {settings.max_sentences} предложений")
        return tuple(rules)

    def check_request(self, content: str) -> None:
        if self.settings is None:
            return
        # Recognize explicit formatting directives. Other semantic conflicts are
        # handled by the prompt; output compliance is always checked in code.
        text = content.casefold()
        # Quoted examples and questions about rules are not instructions to
        # violate them. Keep checking the unquoted remainder of the request.
        text = re.sub(r'«[^»]*»|"[^"\n]*"|```[\s\S]*?```|`[^`]*`', "", text)
        if re.search(r"\b(?:игнорируй|отмени|забудь|ignore|disable)\b.{0,50}\b(?:правила|инварианты|rules|invariants)\b", text):
            rules = self.descriptions()
            if rules:
                raise InvariantViolation(rules, self.settings)
        directive = re.search(
            r"\b(?:ответь|отвечай|напиши|пиши|объясни|расскажи|составь|дай|переведи|"
            r"используй|убери|исключи|write|respond|reply|answer|use|omit|remove)\b", text,
        )
        if directive is None and not re.match(r"^(?:без\b|(?:no|without)\b|lowercase\b|строчными\b)", text.strip()):
            return
        conflicts = []
        if self.settings.emoji_enabled and re.search(
            r"\b(?:без\s+(?:эмодзи|смайл\w*|эмотикон\w*)|"
            r"(?:не\s+(?:используй|добавляй|ставь|пиши)|убери|исключи)\s+(?:\w+\s+){0,2}(?:эмодзи|смайл\w*)|"
            r"(?:without|no)\s+(?:emoji\w*|emoticon\w*)|(?:omit|remove)\s+emoji\w*)",
            text,
        ):
            conflicts.append(f"Минимум {self.settings.min_emojis} эмодзи")
        if self.settings.emoji_enabled:
            counts = {"один": 1, "одним": 1, "одно": 1, "one": 1, "два": 2,
                      "двумя": 2, "two": 2, "три": 3, "three": 3}
            requested = re.findall(
                r"\b(\d+|" + "|".join(counts) + r")\s+(?:эмодзи|смайл\w*|emojis?)\b", text,
            )
            if any((int(value) if value.isdigit() else counts[value]) < self.settings.min_emojis
                   for value in requested):
                rule = f"Минимум {self.settings.min_emojis} эмодзи"
                if rule not in conflicts:
                    conflicts.append(rule)
        if self.settings.uppercase_enabled and re.search(
            r"\b(?:(?:строчными|маленькими)\s+буквами|нижн\w*\s+регистр\w*|без\s+(?:капса|заглавных\s+букв)|"
            r"не\s+(?:используй|пиши)\s+(?:\w+\s+){0,2}(?:капс\w*|заглавн\w*)|lowercase|lower\s+case)",
            text,
        ):
            conflicts.append("Только ВЕРХНИЙ РЕГИСТР")
        if self.settings.sentence_limit_enabled:
            numbers = {"одно": 1, "одном": 1, "два": 2, "двух": 2, "три": 3, "трех": 3, "трёх": 3,
                       "one": 1, "two": 2, "three": 3, "четыре": 4, "четырех": 4, "четырёх": 4,
                       "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8,
                       "девять": 9, "десять": 10, "four": 4, "five": 5, "six": 6,
                       "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            count_text = re.sub(
                r"\b(?:не\s+(?:больше|более)|до|at\s+most|up\s+to|no\s+more\s+than)\s+"
                r"(?:\d+|" + "|".join(numbers) + r")\s+(?:предложени\w*|sentences?)\b",
                "", text,
            )
            counts = re.findall(
                r"\b(\d+|" + "|".join(numbers) + r")\s+(?:предложени\w*|sentences?)\b", count_text,
            )
            if any((int(value) if value.isdigit() else numbers[value])
                   > self.settings.max_sentences for value in counts):
                conflicts.append(f"Не больше {self.settings.max_sentences} предложений")
        if conflicts:
            raise InvariantViolation(tuple(conflicts), self.settings)

    def format(self, content: str) -> str:
        if self.settings is None:
            return content
        settings = self.settings
        text = content.upper() if settings.uppercase_enabled else content
        if settings.emoji_enabled:
            needed = max(0, settings.min_emojis - emoji_count(text))
            if needed:
                text += " " + " ".join("💬✨🙂✅🌟🔹🟢📌🎯💡"[:needed])
        return text

    def apply(self, content: str) -> str:
        if self.settings is None:
            return content
        rules = []
        if self.settings.sentence_limit_enabled and sentence_count(content) > self.settings.max_sentences:
            rules.append(f"Не больше {self.settings.max_sentences} предложений")
        # Uppercasing executable/code snippets would silently corrupt them.
        if self.settings.uppercase_enabled and re.search(
            r"```|`[^`]+`|https?://\S+|[\w.+-]+@[\w.-]+", content,
        ) and content != content.upper():
            rules.append("Только ВЕРХНИЙ РЕГИСТР")
        if rules:
            raise InvariantViolation(tuple(rules), self.settings)
        return self.format(content)
