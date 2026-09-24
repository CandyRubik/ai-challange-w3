from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, Protocol
from zoneinfo import ZoneInfo

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


class HourlyWeatherPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: str
    weather_code: int
    condition: str
    temperature_c: float
    apparent_temperature_c: float
    precipitation_probability_pct: int
    precipitation_mm: float
    wind_speed_kmh: float
    relative_humidity_pct: int


class HourlyWeatherForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str
    country: str
    timezone: str
    latitude: float
    longitude: float
    points: list[HourlyWeatherPoint]
    source: str = "Open-Meteo"
    step_minutes: int = 60


class WeatherScheduleView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    city: str
    forecast_hours: int
    interval_minutes: int
    enabled: bool
    created_at: datetime
    next_run_at: datetime


class WeatherSummaryView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    city: str
    forecast_hours: int
    collected_at: datetime | None
    temperature_min_c: float | None
    temperature_max_c: float | None
    apparent_temperature_min_c: float | None
    apparent_temperature_max_c: float | None
    precipitation_probability_max: int | None
    precipitation_total_mm: float | None
    wind_speed_max_kmh: float | None
    latest_condition: str | None
    source: str | None
    step_minutes: int | None


class WeatherScheduleCancellation(BaseModel):
    schedule_id: str
    cancelled: bool


class WeatherApi(Protocol):
    async def forecast(self, city: str, forecast_days: int) -> WeatherForecast: ...
    async def hourly_forecast(self, city: str, forecast_hours: int) -> HourlyWeatherForecast: ...


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
        try:
            return await self._open_meteo_daily_forecast(city, forecast_days)
        except ToolError:
            if city.strip().casefold() not in {"москва", "moscow"}:
                raise
            hourly = await self._wttr_hourly_forecast(min(forecast_days * 24, 72))
            grouped: dict[str, list[HourlyWeatherPoint]] = {}
            for point in hourly.points:
                grouped.setdefault(point.time[:10], []).append(point)
            days = []
            for offset, (day_text, points) in enumerate(sorted(grouped.items())):
                days.append(WeatherDay(
                    date=day_text,
                    day_offset=offset,
                    relative_day=(
                        "сегодня" if offset == 0
                        else "завтра" if offset == 1
                        else f"через {offset} дня"
                    ),
                    weather_code=points[len(points) // 2].weather_code,
                    condition=points[len(points) // 2].condition,
                    temperature_min_c=min(point.temperature_c for point in points),
                    temperature_max_c=max(point.temperature_c for point in points),
                    precipitation_probability_max=max(
                        point.precipitation_probability_pct for point in points
                    ),
                ))
            if not days:
                raise ToolError("Резервный источник не вернул дневной прогноз")
            return WeatherForecast(
                location=hourly.location, country=hourly.country,
                timezone=hourly.timezone, latitude=hourly.latitude,
                longitude=hourly.longitude, days=days, source=hourly.source,
            )

    async def _open_meteo_daily_forecast(
        self, city: str, forecast_days: int,
    ) -> WeatherForecast:
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

    async def hourly_forecast(
        self, city: str, forecast_hours: int = 24,
    ) -> HourlyWeatherForecast:
        try:
            return await self._open_meteo_hourly_forecast(city, forecast_hours)
        except ToolError:
            if city.strip().casefold() not in {"москва", "moscow"}:
                raise
            return await self._wttr_hourly_forecast(forecast_hours)

    async def _open_meteo_hourly_forecast(
        self, city: str, forecast_hours: int,
    ) -> HourlyWeatherForecast:
        geocoding = await self._get(
            GEOCODING_URL,
            {"name": city, "count": 1, "language": "ru", "format": "json"},
        )
        results = geocoding.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ToolError(f"Город {city!r} не найден")
        location = results[0]
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
                "hourly": (
                    "temperature_2m,apparent_temperature,precipitation_probability,"
                    "precipitation,weather_code,wind_speed_10m,relative_humidity_2m"
                ),
                "forecast_hours": forecast_hours,
                "timezone": "auto",
            },
        )
        hourly = forecast.get("hourly")
        if not isinstance(hourly, dict):
            raise ToolError("Open-Meteo не вернул почасовой прогноз")
        try:
            rows = zip(
                hourly["time"], hourly["weather_code"], hourly["temperature_2m"],
                hourly["apparent_temperature"], hourly["precipitation_probability"],
                hourly["precipitation"], hourly["wind_speed_10m"],
                hourly["relative_humidity_2m"], strict=True,
            )
            points = [
                HourlyWeatherPoint(
                    time=str(time),
                    weather_code=int(code),
                    condition=weather_condition(int(code)),
                    temperature_c=float(temperature),
                    apparent_temperature_c=float(apparent),
                    precipitation_probability_pct=int(probability),
                    precipitation_mm=float(precipitation),
                    wind_speed_kmh=float(wind),
                    relative_humidity_pct=int(humidity),
                )
                for time, code, temperature, apparent, probability, precipitation,
                wind, humidity in rows
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("Open-Meteo вернул неполный почасовой прогноз") from error
        return HourlyWeatherForecast(
            location=location_name,
            country=str(location.get("country", "")),
            timezone=str(forecast.get("timezone") or location.get("timezone", "")),
            latitude=latitude,
            longitude=longitude,
            points=points,
        )

    async def _wttr_hourly_forecast(self, forecast_hours: int) -> HourlyWeatherForecast:
        """Use wttr.in's documented JSON output when Open-Meteo is unreachable.

        wttr exposes three-hour forecast slots, so the response records that
        resolution explicitly instead of presenting the data as hourly.
        """
        try:
            if self._client is not None:
                response = await self._client.get(
                    "https://wttr.is/Moscow", params={"format": "j1"},
                    headers={"User-Agent": "ai-challenge-weather-dashboard/1.0"},
                )
            else:
                async with httpx.AsyncClient(timeout=12) as client:
                    response = await client.get(
                        "https://wttr.is/Moscow", params={"format": "j1"},
                        headers={"User-Agent": "ai-challenge-weather-dashboard/1.0"},
                    )
            response.raise_for_status()
            payload = response.json()
            area = payload["nearest_area"][0]
            area_name = area["areaName"][0]["value"]
            country = area["country"][0]["value"]
            latitude = float(area["latitude"])
            longitude = float(area["longitude"])
            days = payload["weather"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            raise ToolError("Резервный погодный источник временно недоступен") from error

        timezone = ZoneInfo("Europe/Moscow")
        now = datetime.now(timezone)
        end = now + timedelta(hours=forecast_hours)
        condition_names = {
            113: "ясно", 116: "переменная облачность", 119: "облачно",
            122: "пасмурно", 143: "дымка", 176: "возможен дождь",
            179: "возможен снег", 182: "возможен мокрый снег",
            200: "гроза", 227: "метель", 230: "сильная метель",
            248: "туман", 260: "ледяной туман", 263: "морось",
            266: "лёгкая морось", 281: "ледяная морось",
            284: "сильная ледяная морось", 293: "возможен лёгкий дождь",
            296: "лёгкий дождь", 299: "временами дождь",
            302: "умеренный дождь", 305: "временами сильный дождь",
            308: "сильный дождь", 311: "лёгкий ледяной дождь",
            314: "ледяной дождь", 317: "мокрый снег",
            320: "сильный мокрый снег", 323: "возможен лёгкий снег",
            326: "лёгкий снег", 329: "временами снег",
            332: "умеренный снег", 335: "сильный снег",
            338: "сильный снег", 350: "ледяные крупинки",
            353: "небольшой дождь", 356: "ливень", 359: "сильный ливень",
            362: "мокрый снег", 365: "сильный мокрый снег",
            368: "снегопад", 371: "сильный снегопад",
            386: "возможна гроза с дождём", 389: "гроза с дождём",
            392: "возможна гроза со снегом", 395: "гроза со снегом",
        }

        def number(record: dict, key: str, default: float = 0) -> float:
            try:
                return float(record[key])
            except (KeyError, TypeError, ValueError):
                return default

        points: list[HourlyWeatherPoint] = []
        for day in days:
            try:
                forecast_date = date.fromisoformat(day["date"])
                slots = day["hourly"]
            except (KeyError, TypeError, ValueError):
                continue
            for slot in slots:
                try:
                    hour = int(slot["time"]) // 100
                    point_time = datetime.combine(forecast_date, datetime.min.time(), timezone)
                    point_time += timedelta(hours=hour)
                    if point_time < now.replace(minute=0, second=0, microsecond=0) or point_time > end:
                        continue
                    code = int(slot["weatherCode"])
                except (KeyError, TypeError, ValueError):
                    continue
                points.append(HourlyWeatherPoint(
                    time=point_time.isoformat(),
                    weather_code=code,
                    condition=condition_names.get(code, "погодные условия без описания"),
                    temperature_c=number(slot, "tempC"),
                    apparent_temperature_c=number(slot, "FeelsLikeC", number(slot, "tempC")),
                    precipitation_probability_pct=int(max(
                        number(slot, "chanceofrain"), number(slot, "chanceofsnow"),
                        number(slot, "chanceofthunder"),
                    )),
                    precipitation_mm=number(slot, "precipMM"),
                    wind_speed_kmh=number(slot, "windspeedKmph"),
                    relative_humidity_pct=int(number(slot, "humidity")),
                ))
        if not points:
            raise ToolError("Резервный источник не вернул прогноз на ближайшие часы")
        return HourlyWeatherForecast(
            location=str(area_name), country=str(country), timezone="Europe/Moscow",
            latitude=latitude, longitude=longitude, points=points,
            source="wttr.in", step_minutes=180,
        )


def create_weather_server(
    weather_api: WeatherApi | None = None,
    repository: object | None = None,
) -> MCPServer:
    api = weather_api or OpenMeteoWeatherApi()
    if repository is None:
        from ..scheduler.storage import SQLiteWeatherScheduleRepository

        repository = SQLiteWeatherScheduleRepository()
    server = MCPServer(
        "weather",
        instructions=(
            "Use get_weather_forecast for current weather-forecast questions. "
            "Use create_weather_schedule only when the user explicitly requests "
            "periodic collection. It collects hourly weather in the background. "
            "Use get_weather_summary for the latest stored forecast and "
            "cancel_weather_schedule to stop collection. "
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

    @server.tool(title="Создать расписание сбора погоды")
    async def create_weather_schedule(
        city: Annotated[str, Field(min_length=2, max_length=100)],
        interval_minutes: Annotated[
            int,
            Field(ge=1, le=10080, description="Интервал в минутах, от 1 до 10080"),
        ] = 60,
        forecast_hours: Annotated[int, Field(ge=1, le=48)] = 24,
    ) -> WeatherScheduleView:
        """Почасово собирать прогноз для указанного города в фоне."""
        try:
            schedule = repository.create_schedule(city, interval_minutes, forecast_hours)
        except ValueError as error:
            raise ToolError(str(error)) from error
        return WeatherScheduleView.model_validate(schedule)

    @server.tool(title="Список расписаний погоды")
    async def list_weather_schedules() -> list[WeatherScheduleView]:
        """Показать активные и остановленные расписания сбора погоды."""
        return [
            WeatherScheduleView.model_validate(schedule)
            for schedule in repository.list_schedules()
        ]

    @server.tool(title="Сводка сохранённых наблюдений погоды")
    async def get_weather_summary(
        city: Annotated[str, Field(min_length=2, max_length=100)],
    ) -> WeatherSummaryView:
        """Суммировать почасовой прогноз на ближайшие сутки из SQLite."""
        summary = repository.summarize(city)
        return WeatherSummaryView.model_validate(summary)

    @server.tool(title="Остановить расписание погоды")
    async def cancel_weather_schedule(
        schedule_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> WeatherScheduleCancellation:
        """Остановить периодический сбор по идентификатору расписания."""
        cancelled = repository.cancel_schedule(schedule_id)
        if not cancelled:
            raise ToolError("Активное расписание с таким идентификатором не найдено")
        return WeatherScheduleCancellation(schedule_id=schedule_id, cancelled=True)

    return server


mcp = create_weather_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
