from __future__ import annotations

import asyncio
import logging

from ..mcp.weather_server import OpenMeteoWeatherApi
from .storage import SQLiteWeatherScheduleRepository


logger = logging.getLogger("weather_scheduler")
POLL_SECONDS = 5


async def run_scheduler(
    repository: SQLiteWeatherScheduleRepository | None = None,
    weather_api: OpenMeteoWeatherApi | None = None,
    *,
    poll_seconds: int = POLL_SECONDS,
) -> None:
    repository = repository or SQLiteWeatherScheduleRepository()
    weather_api = weather_api or OpenMeteoWeatherApi()
    repository.ensure_default_schedule()
    logger.info("Weather scheduler started; polling every %s seconds", poll_seconds)
    while True:
        for schedule in repository.claim_due():
            try:
                forecast = await weather_api.hourly_forecast(
                    schedule.city, schedule.forecast_hours,
                )
                repository.record_success(schedule, forecast)
                logger.info("Collected weather for %s (schedule %s)", schedule.city, schedule.id)
            except Exception as error:
                repository.record_failure(schedule, str(error))
                logger.exception("Weather collection failed for %s", schedule.city)
        await asyncio.sleep(poll_seconds)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        logger.info("Weather scheduler stopped")


if __name__ == "__main__":
    main()
