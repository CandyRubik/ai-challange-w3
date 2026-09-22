# День 17. Собственный Weather MCP

## Результат

Приложение содержит собственный MCP-сервер `weather`. Он регистрирует
инструмент `get_weather_forecast`, обращается к Open-Meteo и возвращает
типизированный прогноз. Агент сам выбирает инструмент для погодного вопроса,
получает результат через MCP и использует его в финальном ответе.

```text
Веб-чат
  → DeepSeek выбирает tool + arguments
  → MCP Client запускает локальный stdio-процесс
  → Weather MCP вызывает Open-Meteo
  → structuredContent возвращается агенту
  → ответ помечается «MCP · get_weather_forecast»
```

## Контракт инструмента

Инструмент зарегистрирован декоратором `@server.tool()` в
`app/mcp/weather_server.py`.

Вход:

- `city: string` — название города, обязательный параметр;
- `forecast_days: integer` — от 1 до 7 дней, по умолчанию 1.

Выход:

- найденный город, страна, координаты и часовой пояс;
- дата, смещение от сегодня, weather code и человекочитаемое состояние;
- минимальная и максимальная температура;
- максимальная вероятность осадков для каждого дня;
- название источника данных.

MCP SDK строит `inputSchema` и `outputSchema` из type hints и Pydantic-моделей.

## Запуск и проверка

Установить зависимости и запустить приложение:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
set -a
source .env
set +a
uvicorn app.main:app --reload --port 8000
```

Проверить stdio-сервер через HTTP-диагностику приложения. `/api/mcp/tools`
показывает автоматически сгенерированные `input_schema` и `output_schema`:

```bash
curl http://127.0.0.1:8000/api/mcp/status
curl http://127.0.0.1:8000/api/mcp/tools
```

Запустить сам MCP-сервер можно командой:

```bash
.venv/bin/python -m app.mcp.weather_server
```

Она ожидает MCP JSON-RPC в stdin, поэтому для интерактивной проверки удобнее
использовать Inspector:

```bash
.venv/bin/mcp dev app/mcp/weather_server.py
```

Для подключения к Codex CLI из корня проекта:

```bash
codex mcp add challenge-weather -- .venv/bin/python -m app.mcp.weather_server
codex mcp list
```

Само веб-приложение не зависит от пользовательской конфигурации Codex: оно
создаёт `StdioServerParameters`, запускает тот же модуль и вызывает инструмент
через официальный MCP Client.

## Тесты

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

Покрыто:

- регистрация инструмента и автоматически сгенерированная входная схема;
- структурированный MCP-результат;
- адаптер Geocoding API + Forecast API без реальной сети;
- выбор инструмента агентом и валидация аргументов;
- реальный discovery через stdio-процесс;
- передача результата в system context и финальный ответ;
- диагностические API и MCP-маркер в интерфейсе.

## Сценарий видео

1. Показать `@server.tool()` и параметры `city`, `forecast_days`.
2. Открыть `/api/mcp/tools`: видны имя, описание и JSON Schema.
3. В приложении завершить короткий onboarding или отправить `/skip`.
4. Спросить: «Нужен ли завтра зонт в Москве?»
5. Показать ответ с зелёным маркером `MCP · get_weather_forecast`.
6. Коротко показать в результате вероятность осадков и рекомендацию агента.
7. Завершить кадром с успешным запуском тестов.

Open-Meteo используется как внешний API без ключа для некоммерческой
демонстрации; данные требуют атрибуции согласно условиям сервиса.
