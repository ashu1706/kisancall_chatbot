# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from app.nlp_engine import process_query
from app.user_location import set_user_location, get_user_location
from app.location_weather import fetch_weather_for_location  # ✅ updated

app = FastAPI(title="Kisan Call Chatbot API")

# ---------- Request Models ----------
class ChatRequest(BaseModel):
    query: str = Field(..., example="What is the weather in Delhi today?")
    user_id: Optional[str] = Field(None, example="farmer_123")
    lat: Optional[float] = Field(None, example=28.6139)
    lon: Optional[float] = Field(None, example=77.2090)

class LocationUpdate(BaseModel):
    user_id: str = Field(..., example="farmer_123")
    lat: float = Field(..., example=28.6139)
    lon: float = Field(..., example=77.2090)
    location_name: Optional[str] = Field(None, example="Delhi, India")

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Chatbot backend is running"}

@app.post("/update-location")
def update_location(req: LocationUpdate):
    saved = set_user_location(req.user_id, req.lat, req.lon, req.location_name)
    return {"message": "Location updated", "location": saved}

@app.get("/weather")
def get_weather_for_user(user_id: str):
    stored = get_user_location(user_id)
    if not stored:
        return {"error": "Location not set for this user_id"}

    # use fetch_weather_for_location instead of get_weather
    weather_info = fetch_weather_for_location(stored["location_name"] or "")
    return {"user_id": user_id, "location": stored, "weather": weather_info}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    result = process_query(
        user_text=request.query,
        user_id=request.user_id,
        lat=request.lat,
        lon=request.lon
    )
    return result