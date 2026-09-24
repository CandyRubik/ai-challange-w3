from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Protocol
from uuid import uuid4

from ..storage.chat_sessions import DEFAULT_CHAT_DB_PATH


class HourlyPoint(Protocol):
    time: str
    condition: str
    temperature_c: float
    apparent_temperature_c: float
    precipitation_probability_pct: int
    precipitation_mm: float
    wind_speed_kmh: float
    relative_humidity_pct: int


class HourlyForecast(Protocol):
    location: str
    timezone: str
    source: str
    step_minutes: int
    points: list[HourlyPoint]


@dataclass(frozen=True, slots=True)
class WeatherSchedule:
    id: str
    city: str
    forecast_hours: int
    interval_minutes: int
    enabled: bool
    created_at: datetime
    next_run_at: datetime


@dataclass(frozen=True, slots=True)
class WeatherSummary:
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


class SQLiteWeatherScheduleRepository:
    """Store hourly forecasts and schedules in the application's SQLite DB."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self._database_path = Path(
            database_path or os.getenv("CHAT_DB_PATH") or DEFAULT_CHAT_DB_PATH
        )
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS weather_schedules (
                    id TEXT PRIMARY KEY,
                    city TEXT NOT NULL,
                    forecast_hours INTEGER NOT NULL DEFAULT 24
                        CHECK (forecast_hours BETWEEN 1 AND 48),
                    interval_minutes INTEGER NOT NULL CHECK (interval_minutes BETWEEN 1 AND 10080),
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                    created_at TEXT NOT NULL,
                    next_run_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_weather_schedules_due
                ON weather_schedules(enabled, next_run_at);

                CREATE TABLE IF NOT EXISTS weather_runs (
                    id TEXT PRIMARY KEY,
                    schedule_id TEXT NOT NULL REFERENCES weather_schedules(id) ON DELETE CASCADE,
                    city TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'Open-Meteo',
                    step_minutes INTEGER NOT NULL DEFAULT 60,
                    observed_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_weather_runs_city_time
                ON weather_runs(city COLLATE NOCASE, observed_at);

                CREATE TABLE IF NOT EXISTS weather_hourly_observations (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES weather_runs(id) ON DELETE CASCADE,
                    forecast_time TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    temperature_c REAL NOT NULL,
                    apparent_temperature_c REAL NOT NULL,
                    precipitation_probability_pct INTEGER NOT NULL,
                    precipitation_mm REAL NOT NULL,
                    wind_speed_kmh REAL NOT NULL,
                    relative_humidity_pct INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_weather_hourly_run_time
                ON weather_hourly_observations(run_id, forecast_time);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(weather_schedules)")
            }
            if "forecast_hours" not in columns:
                connection.execute(
                    "ALTER TABLE weather_schedules ADD COLUMN forecast_hours INTEGER NOT NULL DEFAULT 24"
                )
            run_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(weather_runs)")
            }
            if "source" not in run_columns:
                connection.execute(
                    "ALTER TABLE weather_runs ADD COLUMN source TEXT NOT NULL DEFAULT 'Open-Meteo'"
                )
            if "step_minutes" not in run_columns:
                connection.execute(
                    "ALTER TABLE weather_runs ADD COLUMN step_minutes INTEGER NOT NULL DEFAULT 60"
                )
            connection.commit()

    @staticmethod
    def _parse_schedule(row: sqlite3.Row) -> WeatherSchedule:
        return WeatherSchedule(
            id=row["id"],
            city=row["city"],
            forecast_hours=row["forecast_hours"],
            interval_minutes=row["interval_minutes"],
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            next_run_at=datetime.fromisoformat(row["next_run_at"]),
        )

    def create_schedule(
        self, city: str, interval_minutes: int = 60, forecast_hours: int = 24,
    ) -> WeatherSchedule:
        normalized_city = city.strip()
        if not normalized_city:
            raise ValueError("Укажите город")
        if not 1 <= interval_minutes <= 10080 or not 1 <= forecast_hours <= 48:
            raise ValueError("Недопустимый интервал или горизонт прогноза")
        now = datetime.now(timezone.utc)
        schedule_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            enabled_count = connection.execute(
                "SELECT COUNT(*) FROM weather_schedules WHERE enabled = 1"
            ).fetchone()[0]
            if enabled_count >= 100:
                raise ValueError("Достигнут лимит в 100 активных расписаний")
            connection.execute(
                """INSERT INTO weather_schedules
                   (id, city, forecast_hours, interval_minutes, enabled, created_at, next_run_at)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (schedule_id, normalized_city, forecast_hours, interval_minutes,
                 now.isoformat(), now.isoformat()),
            )
            connection.commit()
        return WeatherSchedule(
            id=schedule_id,
            city=normalized_city,
            forecast_hours=forecast_hours,
            interval_minutes=interval_minutes,
            enabled=True,
            created_at=now,
            next_run_at=now,
        )

    def ensure_default_schedule(self) -> WeatherSchedule | None:
        now = datetime.now(timezone.utc)
        schedule_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            has_any = connection.execute("SELECT 1 FROM weather_schedules LIMIT 1").fetchone()
            if has_any:
                connection.commit()
                return None
            connection.execute(
                """INSERT INTO weather_schedules
                   (id, city, forecast_hours, interval_minutes, enabled, created_at, next_run_at)
                   VALUES (?, 'Москва', 24, 60, 1, ?, ?)""",
                (schedule_id, now.isoformat(), now.isoformat()),
            )
            connection.commit()
        return WeatherSchedule(
            id=schedule_id, city="Москва", forecast_hours=24, interval_minutes=60,
            enabled=True, created_at=now, next_run_at=now,
        )

    def list_schedules(self) -> list[WeatherSchedule]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM weather_schedules ORDER BY created_at DESC"
            ).fetchall()
        return [self._parse_schedule(row) for row in rows]

    def claim_due(self, now: datetime | None = None) -> list[WeatherSchedule]:
        """Advance due schedules atomically before performing external requests."""
        now = now or datetime.now(timezone.utc)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM weather_schedules
                   WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at""",
                (now.isoformat(),),
            ).fetchall()
            schedules = [self._parse_schedule(row) for row in rows]
            for schedule in schedules:
                next_run = now + timedelta(minutes=schedule.interval_minutes)
                connection.execute(
                    "UPDATE weather_schedules SET next_run_at = ? WHERE id = ?",
                    (next_run.isoformat(), schedule.id),
                )
            connection.commit()
        return schedules

    def record_success(self, schedule: WeatherSchedule, forecast: HourlyForecast) -> None:
        observed_at = datetime.now(timezone.utc).isoformat()
        run_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute(
                """INSERT INTO weather_runs
                   (id, schedule_id, city, timezone, source, step_minutes, observed_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'ok')""",
                (run_id, schedule.id, forecast.location, forecast.timezone,
                 forecast.source, forecast.step_minutes, observed_at),
            )
            connection.executemany(
                """INSERT INTO weather_hourly_observations
                   (id, run_id, forecast_time, condition, temperature_c,
                    apparent_temperature_c, precipitation_probability_pct,
                    precipitation_mm, wind_speed_kmh, relative_humidity_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (str(uuid4()), run_id, point.time, point.condition, point.temperature_c,
                     point.apparent_temperature_c, point.precipitation_probability_pct,
                     point.precipitation_mm, point.wind_speed_kmh,
                     point.relative_humidity_pct)
                    for point in forecast.points
                ],
            )
            connection.commit()

    def record_failure(self, schedule: WeatherSchedule, error: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """INSERT INTO weather_runs
                   (id, schedule_id, city, timezone, observed_at, status, error)
                   VALUES (?, ?, ?, '', ?, 'error', ?)""",
                (str(uuid4()), schedule.id, schedule.city,
                 datetime.now(timezone.utc).isoformat(), error[:500]),
            )
            connection.commit()

    def summarize(self, city: str) -> WeatherSummary:
        with closing(self._connect()) as connection:
            run = connection.execute(
                """SELECT id, observed_at, source, step_minutes FROM weather_runs
                   WHERE status = 'ok' AND city = ? COLLATE NOCASE
                   ORDER BY observed_at DESC LIMIT 1""",
                (city.strip(),),
            ).fetchone()
            aggregate = connection.execute(
                """SELECT COUNT(*) AS forecast_hours,
                          MIN(temperature_c) AS temperature_min_c,
                          MAX(temperature_c) AS temperature_max_c,
                          MIN(apparent_temperature_c) AS apparent_temperature_min_c,
                          MAX(apparent_temperature_c) AS apparent_temperature_max_c,
                          MAX(precipitation_probability_pct) AS precipitation_probability_max,
                          SUM(precipitation_mm) AS precipitation_total_mm,
                          MAX(wind_speed_kmh) AS wind_speed_max_kmh
                   FROM weather_hourly_observations WHERE run_id = ?""",
                (run["id"],) if run else ("",),
            ).fetchone()
            latest = connection.execute(
                """SELECT condition FROM weather_hourly_observations
                   WHERE run_id = ? ORDER BY forecast_time LIMIT 1""",
                (run["id"],) if run else ("",),
            ).fetchone()
        return WeatherSummary(
            city=city.strip(),
            forecast_hours=aggregate["forecast_hours"] or 0,
            collected_at=datetime.fromisoformat(run["observed_at"]) if run else None,
            temperature_min_c=aggregate["temperature_min_c"],
            temperature_max_c=aggregate["temperature_max_c"],
            apparent_temperature_min_c=aggregate["apparent_temperature_min_c"],
            apparent_temperature_max_c=aggregate["apparent_temperature_max_c"],
            precipitation_probability_max=aggregate["precipitation_probability_max"],
            precipitation_total_mm=aggregate["precipitation_total_mm"],
            wind_speed_max_kmh=aggregate["wind_speed_max_kmh"],
            latest_condition=latest["condition"] if latest else None,
            source=run["source"] if run else None,
            step_minutes=run["step_minutes"] if run else None,
        )

    def cancel_schedule(self, schedule_id: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE weather_schedules SET enabled = 0 WHERE id = ? AND enabled = 1",
                (schedule_id,),
            )
            connection.commit()
        return cursor.rowcount > 0

    def overview(self) -> list[dict[str, object]]:
        """Return each schedule with its latest run and hourly forecast rows."""
        result: list[dict[str, object]] = []
        for schedule in self.list_schedules():
            with closing(self._connect()) as connection:
                run = connection.execute(
                    """SELECT id, observed_at, status, error, source, step_minutes FROM weather_runs
                       WHERE schedule_id = ? ORDER BY observed_at DESC LIMIT 1""",
                    (schedule.id,),
                ).fetchone()
                points = []
                if run and run["status"] == "ok":
                    points = [
                        dict(row)
                        for row in connection.execute(
                            """SELECT forecast_time, condition, temperature_c,
                                      apparent_temperature_c, precipitation_probability_pct,
                                      precipitation_mm, wind_speed_kmh, relative_humidity_pct
                               FROM weather_hourly_observations WHERE run_id = ?
                               ORDER BY forecast_time""",
                            (run["id"],),
                        ).fetchall()
                    ]
            result.append(
                {
                    "id": schedule.id,
                    "city": schedule.city,
                    "forecast_hours": schedule.forecast_hours,
                    "interval_minutes": schedule.interval_minutes,
                    "enabled": schedule.enabled,
                    "next_run_at": schedule.next_run_at,
                    "last_run_at": datetime.fromisoformat(run["observed_at"]) if run else None,
                    "last_status": run["status"] if run else None,
                    "last_error": run["error"] if run else None,
                    "last_source": run["source"] if run else None,
                    "step_minutes": run["step_minutes"] if run else None,
                    "forecast": points,
                }
            )
        return result

    def history(self, city: str, hours: int = 72) -> list[dict[str, object]]:
        """Keep one forecast snapshot per hourly collection for a change timeline."""
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with closing(self._connect()) as connection:
            runs = connection.execute(
                """SELECT id, observed_at, timezone, source, step_minutes FROM weather_runs
                   WHERE status = 'ok' AND city = ? COLLATE NOCASE AND observed_at >= ?
                   ORDER BY observed_at DESC LIMIT ?""",
                (city.strip(), since, hours + 1),
            ).fetchall()
            history = []
            for run in runs:
                point = connection.execute(
                    """SELECT forecast_time, condition, temperature_c,
                              apparent_temperature_c, precipitation_probability_pct,
                              precipitation_mm, wind_speed_kmh
                       FROM weather_hourly_observations WHERE run_id = ?
                       ORDER BY forecast_time LIMIT 1""",
                    (run["id"],),
                ).fetchone()
                if point is not None:
                    history.append({
                        "collected_at": datetime.fromisoformat(run["observed_at"]),
                        "timezone": run["timezone"],
                        "source": run["source"],
                        "step_minutes": run["step_minutes"],
                        **dict(point),
                    })
        return list(reversed(history))

    def latest_weather_context(self, city: str) -> str:
        entry = next(
            (
                item for item in self.overview()
                if str(item["city"]).casefold() == city.strip().casefold()
                and item["last_status"] == "ok" and item["forecast"]
            ),
            None,
        )
        if entry is None:
            return f"Нет сохранённого почасового прогноза для города {city.strip()}."
        lines = [
            f"Город: {entry['city']}",
            f"Последнее обновление: {entry['last_run_at'].isoformat()}",
            f"Источник: {entry['last_source']}; шаг прогноза: {entry['step_minutes']} минут.",
            "Время указано локальное для города:",
        ]
        for point in entry["forecast"]:
            lines.append(
                f"{point['forecast_time']}: {point['condition']}; "
                f"температура {point['temperature_c']} °C, ощущается как "
                f"{point['apparent_temperature_c']} °C; осадки "
                f"{point['precipitation_probability_pct']}% / "
                f"{point['precipitation_mm']} мм; ветер "
                f"{point['wind_speed_kmh']} км/ч; влажность "
                f"{point['relative_humidity_pct']}%."
            )
        history = self.history(city, 24)
        if history:
            lines.append("Сохранённые почасовые изменения:")
            for point in history:
                lines.append(
                    f"{point['collected_at'].isoformat()}: "
                    f"{point['temperature_c']} °C, ощущается как "
                    f"{point['apparent_temperature_c']} °C, {point['condition']}, "
                    f"осадки {point['precipitation_probability_pct']}%, "
                    f"ветер {point['wind_speed_kmh']} км/ч."
                )
        return "\n".join(lines)
