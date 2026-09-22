# День 16. Подключение MCP

## Выбранный открытый MCP

Для демонстрации используется официальный публичный **DeepWiki MCP**:

- endpoint: `https://mcp.deepwiki.com/mcp`;
- транспорт: Streamable HTTP;
- авторизация: не требуется для публичных GitHub-репозиториев;
- инструменты: `read_wiki_structure`, `read_wiki_contents` и инструмент вопросов
  (`ask_wiki_question`; прежнее имя `ask_question` также поддержано клиентом).

Документация сервера: <https://docs.devin.ai/work-with-devin/deepwiki-mcp>.

## План реализации

1. Установить официальный Python SDK `mcp`.
2. Открыть Streamable HTTP-соединение с DeepWiki MCP.
3. Дождаться согласования версии протокола и вывести данные соединения.
4. Выполнить MCP-запрос `tools/list`, обработав возможную пагинацию.
5. Вывести имя, описание и входную JSON Schema каждого инструмента.
6. Проверить логику unit-тестом и отдельным живым запуском.

## Установка и запуск в сервисе

Требуется Python 3.11+ и доступ в интернет.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

Откройте <http://127.0.0.1:8000> и используйте MCP прямо в поле чата:

```text
Изучи репозиторий modelcontextprotocol/python-sdk через DeepWiki и объясни,
какие транспорты поддерживает MCP Client.
```

Модель сама получает актуальный каталог инструментов, выбирает подходящий tool,
формирует аргументы по его JSON Schema и передаёт результат в итоговый ответ.
Если MCP не нужен, обычный вопрос обрабатывается без вызова инструмента.

Для диагностики и ручного управления остаются slash-команды:

```text
/mcp-help
/mcp-tools
/deepwiki modelcontextprotocol/python-sdk Как устроен MCP Client?
/mcp-call read_wiki_structure {"repoName":"modelcontextprotocol/python-sdk"}
```

Команды и ответы MCP сохраняются в истории, но не передаются языковой модели и
не попадают в память. Поэтому `/mcp-tools` и прямые вызовы работают без
`DEEPSEEK_API_KEY`.

Проверить подключение можно также через HTTP API:

```bash
curl http://127.0.0.1:8000/api/mcp/status
curl http://127.0.0.1:8000/api/mcp/tools
```

## Отдельный CLI

Тот же сервер можно проверить без запуска веб-приложения:

```bash
python -m app.mcp_client
```

Другой Streamable HTTP MCP endpoint можно передать аргументом:

```bash
python -m app.mcp_client --url https://example.com/mcp
```

или переменной окружения:

```bash
MCP_SERVER_URL=https://example.com/mcp python -m app.mcp_client
```

Успешный запуск сначала печатает `Connected`, endpoint и согласованную версию
протокола, затем количество инструментов и полное описание каждого из них.

## Проверка

```bash
python -m pytest -q tests/test_mcp_client.py
python -m app.mcp_client
```

Unit-тест проверяет открытие контекста клиента, вывод сведений о соединении и
получение всех страниц `tools/list`. Живой запуск подтверждает совместимость с
реальным DeepWiki MCP.

## Сценарий видео

1. Показать зависимость `mcp` в `requirements.txt` и файл `app/mcp_client.py`.
2. Коротко указать endpoint DeepWiki и отсутствие API-ключа.
3. Запустить `python -m pytest -q tests/test_mcp_client.py`.
4. Запустить `uvicorn app.main:app --reload --port 8000` и открыть чат.
5. Отправить `/mcp-tools`, затем вопрос через `/deepwiki` и показать ответы в
   истории чата.
