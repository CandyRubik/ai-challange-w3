from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.providers.deepseek import (
    CONTINUATION_PROMPT, DeepSeekProvider, LlmRequestError, LlmTruncatedResponseError,
)


class FakeCompletions:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def completion(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            ),
        ],
    )


def stream_chunk(
    *,
    reasoning: str = "",
    content: str = "",
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    reasoning_content=reasoning,
                    content=content,
                ),
            ),
        ],
    )


def client(completions: FakeCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_provider_uses_chat_defaults() -> None:
    completions = FakeCompletions(completion("Готово"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    assert provider.generate(messages=[{"role": "user", "content": "Вопрос"}]) == "Готово"
    assert completions.requests == [{
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Вопрос"}],
        "max_tokens": 2_000,
        "stream": False,
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }]


def test_provider_forwards_agent_context() -> None:
    completions = FakeCompletions(completion("Новый ответ"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
        {"role": "user", "content": "Продолжение"},
    ]

    assert provider.generate(messages=messages) == "Новый ответ"
    assert completions.requests[0]["messages"] == messages


def test_json_response_uses_json_mode_and_requested_budget() -> None:
    completions = FakeCompletions(completion('{"passed":true,"report":"Итог","repair_steps":[]}'))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]
    messages = [{"role": "user", "content": "Проверь результат и верни JSON"}]
    answer = provider.generate_json(messages=messages, max_tokens=8_000)
    assert '"passed":true' in answer
    assert completions.requests[0]["response_format"] == {"type": "json_object"}
    assert completions.requests[0]["max_tokens"] == 8_000
    assert completions.requests[0]["messages"] == messages


def test_json_truncation_restarts_instead_of_concatenating_fragments() -> None:
    full_answer = '{"passed":true,"report":"Итог","repair_steps":[]}'
    completions = FakeCompletions(
        completion('{"passed":true,"report":"Оборванный', finish_reason="length"),
        completion(full_answer),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]
    messages = [{"role": "user", "content": "Верни JSON"}]
    assert provider.generate_json(messages=messages, max_tokens=8_000) == full_answer
    assert len(completions.requests) == 2
    assert completions.requests[1]["messages"] == messages
    assert completions.requests[1]["response_format"] == {"type": "json_object"}
    assert completions.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("thinking_enabled", [True, False])
def test_incomplete_json_is_rejected_with_bounded_retries(thinking_enabled: bool) -> None:
    responses = [completion('{"report":"Оборванный', finish_reason="length")] * (2 if thinking_enabled else 1)
    completions = FakeCompletions(*responses)
    provider = DeepSeekProvider(client=client(completions), thinking_enabled=thinking_enabled)  # type: ignore[arg-type]
    with pytest.raises(LlmTruncatedResponseError):
        provider.generate_json(messages=[{"role": "user", "content": "Верни JSON"}])
    assert len(completions.requests) == len(responses)


def test_empty_json_retry_preserves_json_mode() -> None:
    completions = FakeCompletions(completion(""), completion('{"report":"Итог"}'))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]
    assert provider.generate_json(messages=[{"role": "user", "content": "Верни JSON"}]) == '{"report":"Итог"}'
    assert all(request["response_format"] == {"type": "json_object"} for request in completions.requests)


def test_provider_failure_logs_metadata_without_raw_upstream_error(caplog) -> None:
    completions = FakeCompletions(ValueError("PRIVATE_UPSTREAM_BODY"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]
    with pytest.raises(LlmRequestError):
        provider.generate_json(messages=[{"role": "user", "content": "Верни JSON"}])
    assert "type=ValueError" in caplog.text
    assert "PRIVATE_UPSTREAM_BODY" not in caplog.text


def test_provider_continues_an_answer_cut_by_token_limit() -> None:
    completions = FakeCompletions(
        completion("Первая часть оборвалась на полуслове вер", finish_reason="length"),
        completion("оятностей. Вторая часть завершена."),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    answer = provider.generate(messages=[{"role": "user", "content": "Длинный ответ"}])

    assert answer == (
        "Первая часть оборвалась на полуслове вероятностей. "
        "Вторая часть завершена."
    )
    assert len(completions.requests) == 2
    assert completions.requests[1]["messages"][-2:] == [
        {
            "role": "assistant",
            "content": "Первая часть оборвалась на полуслове вер",
        },
        {"role": "user", "content": CONTINUATION_PROMPT},
    ]


def test_agent_provider_can_disable_thinking() -> None:
    completions = FakeCompletions(completion("Ответ"))
    provider = DeepSeekProvider(
        client=client(completions),  # type: ignore[arg-type]
        model="deepseek-v4-pro",
        thinking_enabled=False,
    )

    provider.generate(messages=[{"role": "user", "content": "Вопрос"}], max_tokens=512)

    request = completions.requests[0]
    assert request["model"] == "deepseek-v4-pro"
    assert request["max_tokens"] == 512
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in request


def test_provider_retries_empty_response_without_thinking() -> None:
    completions = FakeCompletions(
        completion(""),
        completion("Ответ после повтора"),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    assert provider.generate(messages=[{"role": "user", "content": "Вопрос"}]) == "Ответ после повтора"
    assert len(completions.requests) == 2
    assert completions.requests[1]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
    assert "reasoning_effort" not in completions.requests[1]


def test_provider_does_not_retry_content_filtered_response() -> None:
    completions = FakeCompletions(completion("", finish_reason="content_filter"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    with pytest.raises(LlmRequestError, match="finish_reason=content_filter"):
        provider.generate(messages=[{"role": "user", "content": "Вопрос"}])

    assert len(completions.requests) == 1


def test_provider_streams_reasoning_and_content() -> None:
    completions = FakeCompletions(
        iter(
            [
                stream_chunk(reasoning="Сначала проверю факты. "),
                stream_chunk(content="Итоговый ответ.", finish_reason="stop"),
            ],
        ),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    chunks = list(provider.stream(system_prompt="system", user_prompt="user"))

    assert [chunk.reasoning for chunk in chunks if chunk.reasoning] == [
        "Сначала проверю факты. ",
    ]
    assert [chunk.content for chunk in chunks if chunk.content] == [
        "Итоговый ответ.",
    ]
    assert completions.requests[0]["stream"] is True


def test_provider_retries_empty_stream_without_thinking() -> None:
    completions = FakeCompletions(
        iter(
            [
                stream_chunk(reasoning="Думаю, но финала пока нет."),
                stream_chunk(finish_reason="stop"),
            ],
        ),
        iter([stream_chunk(content="Ответ после повтора.", finish_reason="stop")]),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    chunks = list(provider.stream(system_prompt="system", user_prompt="user"))

    assert any(chunk.status for chunk in chunks)
    assert "Ответ после повтора." in "".join(chunk.content for chunk in chunks)
    assert len(completions.requests) == 2
    assert completions.requests[1]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
