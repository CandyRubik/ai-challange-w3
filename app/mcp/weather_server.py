from __future__ import annotations

from typing import Annotated, Protocol

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherDay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    day_offset: int = Field(description="0 означает сегодня, 1 — завтра")
    relative_day: str
    weather_code: int
    condition: str
    temperature_min_c: float
    temperature_max_c: float
    precipitation_probability_max: int


class WeatherForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str
    country: str
    timezone: str
    latitude: float
    longitude: float
    days: list[WeatherDay]
    source: str = "Open-Meteo"


class WeatherApi(Protocol):
    async def forecast(self, city: str, forecast_days: int) -> WeatherForecast: ...


def weather_condition(code: int) -> str:
    if code == 0:
        return "ясно"
    if code in {1, 2, 3}:
        return "облачно"
    if code in {45, 48}:
        return "туман"
    if code in {51, 53, 55, 56, 57}:
        return "морось"
    if code in {61, 63, 65, 66, 67}:
        return "дождь"
    if code in {71, 73, 75, 77}:
        return "снег"
    if code in {80, 81, 82}:
        return "ливень"
    if code in {85, 86}:
        return "снегопад"
    if code in {95, 96, 99}:
        return "гроза"
    return "неизвестно"


class OpenMeteoWeatherApi:
    """Translate a city name into a compact Open-Meteo daily forecast."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def _get(self, url: str, params: dict[str, str | int | float]) -> dict:
        try:
            if self._client is not None:
                response = await self._client.get(url, params=params)
            else:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ToolError("Open-Meteo временно недоступен") from error
        if not isinstance(payload, dict):
            raise ToolError("Open-Meteo вернул неожиданный формат данных")
        return payload

    async def forecast(self, city: str, forecast_days: int) -> WeatherForecast:
        geocoding = await self._get(
            GEOCODING_URL,
            {"name": city, "count": 1, "language": "ru", "format": "json"},
        )
        results = geocoding.get("results")
        if not isinstance(results, list) or not results:
            raise ToolError(f"Город {city!r} не найден")
        location = results[0]
        if not isinstance(location, dict):
            raise ToolError("Геокодер вернул неожиданный формат данных")

        try:
            latitude = float(location["latitude"])
            longitude = float(location["longitude"])
            location_name = str(location["name"])
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("Геокодер не вернул координаты города") from error

        forecast = await self._get(
            FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": forecast_days,
            },
        )
        daily = forecast.get("daily")
        if not isinstance(daily, dict):
            raise ToolError("Open-Meteo не вернул дневной прогноз")

        try:
            rows = zip(
                daily["time"],
                daily["weather_code"],
                daily["temperature_2m_min"],
                daily["temperature_2m_max"],
                daily["precipitation_probability_max"],
                strict=True,
            )
            days = [
                WeatherDay(
                    date=str(date),
                    day_offset=day_offset,
                    relative_day=(
                        "сегодня" if day_offset == 0
                        else "завтра" if day_offset == 1
                        else f"через {day_offset} дня"
                    ),
                    weather_code=int(code),
                    condition=weather_condition(int(code)),
                    temperature_min_c=float(minimum),
                    temperature_max_c=float(maximum),
                    precipitation_probability_max=int(precipitation),
                )
                for day_offset, (date, code, minimum, maximum, precipitation) in enumerate(rows)
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("Open-Meteo вернул неполный прогноз") from error

        return WeatherForecast(
            location=location_name,
            country=str(location.get("country", "")),
            timezone=str(forecast.get("timezone") or location.get("timezone", "")),
            latitude=latitude,
            longitude=longitude,
            days=days,
        )


def create_weather_server(weather_api: WeatherApi | None = None) -> MCPServer:
    api = weather_api or OpenMeteoWeatherApi()
    server = MCPServer(
        "weather",
        instructions=(
            "Use get_weather_forecast for current weather-forecast questions. "
            "The free Open-Meteo endpoint is intended for non-commercial demo use."
        ),
    )

    @server.tool(title="Прогноз погоды")
    async def get_weather_forecast(
        city: Annotated[
            str,
            Field(
                min_length=2,
                max_length=100,
                description="Название города, при необходимости вместе со страной",
            ),
        ],
        forecast_days: Annotated[
            int,
            Field(
                ge=1,
                le=7,
                description="Количество дней прогноза от сегодняшнего дня",
            ),
        ] = 1,
    ) -> WeatherForecast:
        """Получить прогноз погоды для города через Open-Meteo."""
        return await api.forecast(city, forecast_days)

    return server


mcp = create_weather_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
