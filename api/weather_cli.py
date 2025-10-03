#!/usr/bin/env python3
# api/weather_cli.py
"""
Day 3 - Weather CLI (APIs & JSON basics)

Usage:
    python api/weather_cli.py --city "Mumbai"
    python api/weather_cli.py --city "London"
"""

import sys
import argparse
import requests
from datetime import date

# Mapping of WMO weather codes to human-readable descriptions
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Drizzle (light)", 53: "Drizzle (moderate)", 55: "Drizzle (dense)",
    56: "Freezing drizzle (light)", 57: "Freezing drizzle (dense)",
    61: "Rain (slight)", 63: "Rain (moderate)", 65: "Rain (heavy)",
    66: "Freezing rain (light)", 67: "Freezing rain (heavy)",
    71: "Snow fall (slight)", 73: "Snow fall (moderate)", 75: "Snow fall (heavy)",
    77: "Snow grains",
    80: "Rain showers (slight)", 81: "Rain showers (moderate)", 82: "Rain showers (violent)",
    85: "Snow showers (slight)", 86: "Snow showers (heavy)",
    95: "Thunderstorm (slight/moderate)",
    96: "Thunderstorm + hail (slight)",
    99: "Thunderstorm + hail (heavy)"
}

def geocode_city(name: str):
    """Get latitude/longitude for a city using Open-Meteo's geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    results = data.get("results") or []
    if not results:
        return None
    top = results[0]
    return {
        "name": top.get("name"),
        "country": top.get("country"),
        "lat": top.get("latitude"),
        "lon": top.get("longitude"),
        "timezone": top.get("timezone"),
    }

def fetch_weather(lat: float, lon: float):
    """Fetch current weather and today's forecast for given coordinates."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def describe_code(code: int) -> str:
    """Convert WMO weather code into human-readable string."""
    return WMO_CODES.get(code, f"Code {code}")

def pick_today_daily(daily):
    """Select today's row from the daily forecast array."""
    dates = daily.get("time") or []
    today_iso = str(date.today())
    try:
        idx = dates.index(today_iso)
    except ValueError:
        idx = 0 if dates else None
    if idx is None:
        return None
    return {
        "tmax": (daily.get("temperature_2m_max") or [None])[idx],
        "tmin": (daily.get("temperature_2m_min") or [None])[idx],
        "precip_sum": (daily.get("precipitation_sum") or [None])[idx],
        "date": (dates or [None])[idx],
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Print today's weather for a city.")
    parser.add_argument("--city", required=False, default="Mumbai", help="City name (default: Mumbai)")
    args = parser.parse_args(argv)

    try:
        place = geocode_city(args.city)
        if not place:
            print(f"❌ Could not find city: {args.city}")
            sys.exit(1)

        wx = fetch_weather(place["lat"], place["lon"])
        current = (wx.get("current") or {})
        daily = (wx.get("daily") or {})
        today = pick_today_daily(daily) or {}

        temp = current.get("temperature_2m")
        rh = current.get("relative_humidity_2m")
        wcode = current.get("weather_code")
        wind = current.get("wind_speed_10m")
        precip = current.get("precipitation")

        # Build headline
        headline = (
            f"{place['name']}, {place['country']}: "
            f"{temp}°C, {describe_code(int(wcode) if wcode is not None else -1)}"
        )

        # Build extras
        extras = []
        if rh is not None: extras.append(f"RH {rh}%")
        if wind is not None: extras.append(f"wind {wind} km/h")
        if precip is not None: extras.append(f"precip {precip} mm")
        if today and today.get("tmax") is not None and today.get("tmin") is not None:
            extras.append(f"today {today['tmin']}–{today['tmax']}°C")
        if today and today.get("precip_sum") is not None:
            extras.append(f"daily precip {today['precip_sum']} mm")

        print(headline + (" · " + " · ".join(extras) if extras else ""))

    except requests.exceptions.RequestException as e:
        print(f"❌ Network/API error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
