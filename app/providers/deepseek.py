from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import logging
import os
from typing import Any

from openai import OpenAI

from ..agents.agent import AgentMessage


logger = logging.getLogger(__name__)


class LlmConfigurationError(RuntimeError):
    """The provider cannot be called because local configuration is missing."""


class LlmRequestError(RuntimeError):
    """The provider rejected or failed to complete a request."""


class LlmEmptyStreamError(LlmRequestError):
    """The provider finished a stream without a visible answer."""

    def __init__(self, finish_reason: str | None = None) -> None:
        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        super().__init__(f"DeepSeek вернул пустой потоковый ответ{reason}")
        self.finish_reason = finish_reason


class LlmTruncatedResponseError(LlmRequestError):
    """A structured response did not fit into a single complete answer."""

    def __init__(self) -> None:
        super().__init__("DeepSeek обрезал JSON по лимиту ответа. Полная проверка не получена")


DEFAULT_MAX_TOKENS = 2_000
DEFAULT_REASONING_EFFORT = "high"
MAX_RESPONSE_SEGMENTS = 4
CONTINUATION_PROMPT = (
    "Continue the previous answer exactly from its final character. "
    "Do not repeat any text, add a new introduction, or mention the output limit."
)


@dataclass(frozen=True, slots=True)
class LlmStreamChunk:
    reasoning: str = ""
    content: str = ""
    finish_reason: str | None = None
    status: str = ""


class DeepSeekProvider:
    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
        thinking_enabled: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._thinking_enabled = thinking_enabled

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise LlmConfigurationError("DEEPSEEK_API_KEY не задан")

        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def _build_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
    ) -> dict[str, Any]:
        return self._build_chat_request(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
            stream=stream,
        )

    def _build_chat_request(
        self,
        *,
        messages: Sequence[AgentMessage],
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": list(messages),
            "max_tokens": max_tokens,
            "stream": stream,
            "extra_body": {"thinking": {"type": thinking_type}},
        }
        if thinking_type == "enabled":
            request["reasoning_effort"] = DEFAULT_REASONING_EFFORT
        if response_format is not None:
            request["response_format"] = response_format
        return request

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        request_messages = list(messages)
        parts: list[str] = []
        for segment_index in range(MAX_RESPONSE_SEGMENTS):
            content, finish_reason = self._generate_segment(
                messages=request_messages,
                max_tokens=max_tokens,
            )
            parts.append(content)
            if finish_reason != "length":
                return "".join(parts).strip()
            if segment_index == MAX_RESPONSE_SEGMENTS - 1:
                break
            request_messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": CONTINUATION_PROMPT},
                ],
            )
        raise LlmRequestError(
            "Ответ модели не завершился после нескольких продолжений",
        )

    def _generate_segment(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int,
    ) -> tuple[str, str | None]:
        thinking_type = "enabled" if self._thinking_enabled else "disabled"
        request = self._build_chat_request(
            messages=messages,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
        )
        response = self._request_completion(request)
        content, finish_reason = self._extract_content(response)
        if content:
            return content, finish_reason

        if thinking_type == "enabled" and finish_reason != "content_filter":
            fallback = self._build_chat_request(
                messages=messages,
                thinking_type="disabled",
                max_tokens=max_tokens,
            )
            content, finish_reason = self._extract_content(
                self._request_completion(fallback),
            )
            if content:
                return content, finish_reason

        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise LlmRequestError(f"DeepSeek вернул пустой ответ{reason}")

    def generate_json(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        thinking_types = ("enabled", "disabled") if self._thinking_enabled else ("disabled",)
        finish_reason = None
        for thinking_type in thinking_types:
            request = self._build_chat_request(
                messages=messages, max_tokens=max_tokens,
                response_format={"type": "json_object"}, thinking_type=thinking_type,
            )
            content, finish_reason = self._extract_content(self._request_completion(request))
            if finish_reason == "length":
                # Restart from the original snapshot. JSON mode produces a
                # whole object, so concatenating continuations can corrupt it.
                logger.warning("Structured response truncated (thinking=%s max_tokens=%d)", thinking_type, max_tokens)
                continue
            if content:
                return content
            if finish_reason == "content_filter":
                break
        if finish_reason == "length":
            raise LlmTruncatedResponseError()
        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise LlmRequestError(f"DeepSeek вернул пустой JSON-ответ{reason}")

    def _request_completion(self, request: dict[str, Any]) -> Any:
        try:
            return self._get_client().chat.completions.create(**request)
        except LlmConfigurationError:
            raise
        except Exception as error:
            logger.warning(
                "DeepSeek request failed: type=%s status=%s",
                type(error).__name__, getattr(error, "status_code", None),
            )
            raise LlmRequestError("Запрос к DeepSeek завершился ошибкой") from error

    @staticmethod
    def _extract_content(response: Any) -> tuple[str, str | None]:
        if not response.choices:
            raise LlmRequestError("DeepSeek не вернул вариантов ответа")

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        return content, getattr(choice, "finish_reason", None)

    @staticmethod
    def _read(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _extract_stream_chunk(cls, chunk: Any) -> LlmStreamChunk:
        choices = cls._read(chunk, "choices", []) or []
        if not choices:
            return LlmStreamChunk()

        choice = choices[0]
        delta = cls._read(choice, "delta")
        if delta is None:
            return LlmStreamChunk(
                finish_reason=cls._read(choice, "finish_reason"),
            )

        reasoning = cls._read(delta, "reasoning_content", "")
        if not reasoning:
            reasoning = cls._read(delta, "reasoning", "")
        content = cls._read(delta, "content", "")
        return LlmStreamChunk(
            reasoning=reasoning if isinstance(reasoning, str) else str(reasoning or ""),
            content=content if isinstance(content, str) else str(content or ""),
            finish_reason=cls._read(choice, "finish_reason"),
        )

    def _stream_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        thinking_type: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[LlmStreamChunk]:
        request = self._build_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            response = self._get_client().chat.completions.create(**request)
            saw_content = False
            finish_reason: str | None = None
            for chunk in response:
                parsed = self._extract_stream_chunk(chunk)
                saw_content = saw_content or bool(parsed.content)
                finish_reason = parsed.finish_reason or finish_reason
                if parsed.reasoning or parsed.content or parsed.finish_reason:
                    yield parsed
        except LlmConfigurationError:
            raise
        except LlmEmptyStreamError:
            raise
        except Exception as error:
            raise LlmRequestError("Потоковый запрос к DeepSeek завершился ошибкой") from error

        if not saw_content:
            raise LlmEmptyStreamError(finish_reason)

    def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[LlmStreamChunk]:
        try:
            yield from self._stream_once(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                thinking_type="enabled",
                max_tokens=max_tokens,
            )
        except LlmEmptyStreamError as error:
            if error.finish_reason == "content_filter":
                raise

            yield LlmStreamChunk(
                status="Финальный ответ не пришёл в thinking-режиме; повторяем без thinking…",
            )
            yield from self._stream_once(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                thinking_type="disabled",
                max_tokens=max_tokens,
            )

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        request = self._build_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            thinking_type="enabled",
            max_tokens=max_tokens,
        )
        response = self._request_completion(request)
        content, finish_reason = self._extract_content(response)
        if content:
            return content

        if finish_reason != "content_filter":
            fallback_request = self._build_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                thinking_type="disabled",
                max_tokens=max_tokens,
            )
            fallback_response = self._request_completion(fallback_request)
            fallback_content, fallback_finish_reason = self._extract_content(
                fallback_response,
            )
            if fallback_content:
                return fallback_content
            finish_reason = fallback_finish_reason or finish_reason

        reason = f" (finish_reason={finish_reason})" if finish_reason else ""
        raise LlmRequestError(f"DeepSeek вернул пустой ответ{reason}")
