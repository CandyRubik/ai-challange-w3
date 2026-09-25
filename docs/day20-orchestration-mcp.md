# День 20. Orchestration MCP

## Результат

В host-приложении зарегистрированы два MCP-сервера:

- `weather` — прогноз, расписания и сводка сохранённой погоды;
- `checklist` — создание чек-листа, добавление пунктов и финальная проверка.

Агент видит общий каталог с квалифицированными именами инструментов
(`weather.get_weather_forecast`, `checklist.create_checklist`), выбирает один
следующий вызов, получает его результат и принимает следующее решение. Каждый
вызов проходит JSON Schema validation. Поток ограничен восемью шагами.

```text
Запрос пользователя
  ↓
MCP orchestration router
  ├─ 1 weather.get_weather_forecast
  ├─ 2 checklist.create_checklist
  ├─ 3 checklist.add_checklist_item
  ├─ 4 checklist.add_checklist_item
  └─ 5 checklist.list_checklist_items
  ↓
Финальный ответ агента с полной трассой
```

## Длинный сценарий

Запрос для видео:

> Проверь погоду в Москве на завтра и подготовь чек-лист для прогулки. Если будет
> дождь, добавь зонт; добавь подходящую одежду и покажи готовый список.

Ожидаемый порядок:

1. `weather.get_weather_forecast(city="Москва", forecast_days=2)` получает
   структурированный прогноз.
2. `checklist.create_checklist(title="Прогулка по Москве")` создаёт запись и
   возвращает `checklist_id`.
3. `checklist.add_checklist_item` добавляет зонт, потому что вероятность дождя
   пришла из шага 1.
4. Второй `checklist.add_checklist_item` добавляет одежду с учётом температуры.
5. `checklist.list_checklist_items` возвращает полный список и проверяет результат.
6. Роутер возвращает `{"tool": null, "arguments": {}}`, после чего основной агент
   формирует ответ по трассе.

Состояние Checklist MCP хранится в SQLite. Это нужно, потому что stdio-клиент
создаёт отдельный процесс для каждого вызова, а `checklist_id` должен пережить
переход между вызовами.

## Проверка

```bash
source .venv/bin/activate
PYTHONPATH=. .venv/bin/pytest -q tests/test_mcp_orchestration.py
PYTHONPATH=. .venv/bin/pytest -q
```

Тесты проверяют:

- регистрацию и типизированные схемы Checklist MCP;
- сохранение состояния между `create`, `add` и `list`;
- выбор инструментов разных серверов;
- точный порядок вызовов и передачу `checklist_id` из предыдущего результата;
- остановку потока после `tool=null`.

Для ручной проверки после запуска приложения:

```bash
curl http://127.0.0.1:8000/api/mcp/status
curl http://127.0.0.1:8000/api/mcp/tools
```

В `/api/mcp/tools` должны быть инструменты обоих серверов, а в ответе чата первая
строка будет `Источник: MCP · multi-server flow`.

## Сценарий видео

1. Показать `McpService.default_servers()` и два stdio endpoint-а.
2. Открыть `/api/mcp/tools`: выделить `weather.*` и `checklist.*`, их описания и
   JSON Schema.
3. Открыть `app/services/mcp.py` и показать `run_flow`, `MCP_MAX_FLOW_STEPS` и
   `call_server_tool`.
4. Отправить запрос про прогноз и чек-лист.
5. На экране терминала или в debugger показать последовательность из пяти вызовов:
   сначала Weather MCP, затем Checklist MCP.
6. Показать, что `checklist_id` из ответа `create_checklist` используется в двух
   следующих вызовах.
7. Показать ответ с готовым списком и маркером `MCP · multi-server flow`.
8. Завершить видео выводом `3 passed` для нового тестового файла и общим `pytest -q`.

## Структура кода

- `app/mcp/weather_server.py` — Weather MCP;
- `app/mcp/checklist_server.py` — Checklist MCP и SQLite repository;
- `app/services/mcp.py` — реестр серверов, discovery, schema validation и flow;
- `app/services/chat_sessions.py` — передача полной трассы в финальный ответ;
- `tests/test_mcp_orchestration.py` — сценарий с двумя fake MCP-серверами.
