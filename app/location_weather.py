# app/location_weather.py
import requests
from typing import Tuple, Dict, Any

# Open-Meteo endpoints (no API key needed)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

def get_coordinates(location_name: str) -> Tuple[float, float] | Tuple[None, None]:
    """Convert location name to latitude & longitude using Open-Meteo Geocoding."""
    try:
        params = {"name": location_name, "count": 1, "language": "en"}
        resp = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                lat = data["results"][0]["latitude"]
                lon = data["results"][0]["longitude"]
                return lat, lon
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None


def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch current & recent weather for given coordinates (Open-Meteo)."""
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "windspeed_10m",
                "soil_moisture_0_1cm"
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum"
            ],
            "timezone": "auto"
        }
        response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            result = {}

            # Current weather
            if "current_weather" in data:
                result["current"] = data["current_weather"]

            # Recent hourly trends (last 24h)
            if "hourly" in data:
                result["recent"] = {
                    "temperature": data["hourly"]["temperature_2m"][-24:],
                    "humidity": data["hourly"]["relative_humidity_2m"][-24:],
                    "rainfall": data["hourly"]["rain"][-24:],
                    "windspeed": data["hourly"]["windspeed_10m"][-24:]
                }

            # Daily aggregates
            if "daily" in data:
                result["daily"] = data["daily"]

            return result
        return {"error": "Weather data not available"}
    except Exception as e:
        return {"error": str(e)}



def fetch_weather_for_location(location_name: str) -> Dict[str, Any]:
    """Main helper: takes a location name and returns detailed weather info object."""
    lat, lon = get_coordinates(location_name)
    if not lat or not lon:
        return {"error": "Unable to find location"}
    weather = get_weather(lat, lon)
    return {
        "location": location_name,
        "latitude": lat,
        "longitude": lon,
        "weather": weather
    }