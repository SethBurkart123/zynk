"""
Weather Module

Demonstrates streaming/channel support with simulated weather data.
"""

import asyncio
import random

from pydantic import BaseModel

from zynk import Channel, command


class WeatherData(BaseModel):
    """Current weather conditions."""
    city: str
    temperature: float  # Celsius
    humidity: float  # Percentage
    conditions: str
    wind_speed: float  # km/h


class WeatherForecast(BaseModel):
    """Weather forecast for a specific day."""
    day: str
    high: float
    low: float
    conditions: str
    precipitation_chance: float


class WeatherUpdate(BaseModel):
    """A streaming weather update."""
    timestamp: str
    city: str
    temperature: float
    conditions: str


# Simulated weather data
_weather_conditions = ["Sunny", "Cloudy", "Rainy", "Stormy", "Foggy", "Snowy"]
_city_base_temps = {
    "New York": 15,
    "Los Angeles": 22,
    "Chicago": 10,
    "Miami": 28,
    "Seattle": 12,
    "Denver": 8,
    "Tokyo": 18,
    "London": 11,
    "Paris": 14,
    "Sydney": 23,
}


def _get_simulated_weather(city: str) -> WeatherData:
    """Generate simulated weather for a city."""
    base_temp = _city_base_temps.get(city, 15)
    temp_variation = random.uniform(-5, 5)

    return WeatherData(
        city=city,
        temperature=round(base_temp + temp_variation, 1),
        humidity=round(random.uniform(30, 90), 1),
        conditions=random.choice(_weather_conditions),
        wind_speed=round(random.uniform(0, 30), 1),
    )


@command
async def get_weather(city: str) -> WeatherData:
    """
    Get current weather for a city.

    Returns simulated weather data.
    """
    if city not in _city_base_temps:
        # Still return weather, just with default base temp
        pass

    return _get_simulated_weather(city)


@command
async def get_forecast(city: str, days: int = 7) -> list[WeatherForecast]:
    """
    Get weather forecast for a city.

    Returns a list of forecasts for the specified number of days.
    """
    days = min(days, 14)  # Cap at 14 days
    base_temp = _city_base_temps.get(city, 15)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    forecasts = []

    for i in range(days):
        day = day_names[i % 7]
        high = round(base_temp + random.uniform(0, 10), 1)
        low = round(base_temp - random.uniform(0, 8), 1)

        forecasts.append(WeatherForecast(
            day=day,
            high=high,
            low=low,
            conditions=random.choice(_weather_conditions),
            precipitation_chance=round(random.uniform(0, 100), 0),
        ))

    return forecasts


@command
async def list_cities() -> list[str]:
    """Get list of available cities with weather data."""
    return list(_city_base_temps.keys())


@command
async def stream_weather(city: str, interval_seconds: float, channel: Channel[WeatherUpdate]) -> None:
    """
    Stream live weather updates for a city.

    This demonstrates Zynk's streaming/channel support.
    Updates are sent at the specified interval until the channel is closed.

    Args:
        city: The city to monitor.
        interval_seconds: How often to send updates (1-60 seconds).
        channel: The streaming channel (automatically injected).
    """
    from datetime import datetime

    # Validate interval
    interval = max(1.0, min(60.0, interval_seconds))

    # Send updates until channel is closed
    update_count = 0
    max_updates = 100  # Safety limit

    while update_count < max_updates:
        weather = _get_simulated_weather(city)

        update = WeatherUpdate(
            timestamp=datetime.now().isoformat(),
            city=city,
            temperature=weather.temperature,
            conditions=weather.conditions,
        )

        try:
            await channel.send(update)
            update_count += 1
        except RuntimeError:
            # Channel was closed
            break

        await asyncio.sleep(interval)


@command
async def stream_multi_city(cities: list[str], channel: Channel[WeatherUpdate]) -> None:
    """
    Stream weather updates for multiple cities.

    Cycles through the provided cities, sending one update every 2 seconds.
    """
    from datetime import datetime

    if not cities:
        cities = list(_city_base_temps.keys())[:3]

    update_count = 0
    max_updates = 50
    city_index = 0

    while update_count < max_updates:
        city = cities[city_index % len(cities)]
        weather = _get_simulated_weather(city)

        update = WeatherUpdate(
            timestamp=datetime.now().isoformat(),
            city=city,
            temperature=weather.temperature,
            conditions=weather.conditions,
        )

        try:
            await channel.send(update)
            update_count += 1
            city_index += 1
        except RuntimeError:
            break

        await asyncio.sleep(2.0)
