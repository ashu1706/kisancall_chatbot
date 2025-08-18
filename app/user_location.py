# app/user_location_store.py
from typing import Dict, Optional

# Simple in-memory store. Replace with DB later.
_USER_LOCATIONS: Dict[str, dict] = {}

def set_user_location(user_id: str, lat: float, lon: float, location_name: Optional[str] = None) -> dict:
    _USER_LOCATIONS[user_id] = {"lat": lat, "lon": lon, "location_name": location_name}
    return _USER_LOCATIONS[user_id]

def get_user_location(user_id: str) -> Optional[dict]:
    return _USER_LOCATIONS.get(user_id)
