# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from app.nlp_engine import process_query
from app.user_location import set_user_location, get_user_location
from app.location_weather import fetch_weather_for_location
from app.fertilizer_services import get_fertilizer_recommendation
from app.user_polygons import set_polygon, get_polygon
from fastapi import FastAPI, File, UploadFile
import shutil, os

app = FastAPI(title="Kisan Call Chatbot API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "filename": file.filename, "path": file_path}

@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "filename": file.filename, "path": file_path}
# ---------- Request Models ----------
class ChatRequest(BaseModel):
    query: str = Field(..., example="What is the weather in Delhi today?")
    user_id: Optional[str] = Field(None, example="farmer_123")
    lat: Optional[float] = Field(None, example=28.6139)
    lon: Optional[float] = Field(None, example=77.2090)
    crop: Optional[str] = Field(None, example="wheat")   # ✅ for fertilizer queries

class LocationUpdate(BaseModel):
    user_id: str = Field(..., example="farmer_123")
    lat: float = Field(..., example=28.6139)
    lon: float = Field(..., example=77.2090)
    location_name: Optional[str] = Field(None, example="Delhi, India")

class PolygonRequest(BaseModel):
    user_id: str
    polygon_id: str

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

    weather_info = fetch_weather_for_location(stored["location_name"] or "")
    return {"user_id": user_id, "location": stored, "weather": weather_info}

@app.post("/set-polygon")
def set_user_polygon(req: PolygonRequest):
    """
    Save a polygon_id for a given user_id (after farm registration).
    """
    polygon_id = set_polygon(req.user_id, req.polygon_id)
    return {
        "message": "Polygon linked successfully",
        "user_id": req.user_id,
        "polygon_id": polygon_id
    }
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Main chatbot endpoint: handles weather, fertilizer, and general queries.
    """
    query = request.query.lower()

    # ✅ Fertilizer intent detection
    if any(word in query for word in ["fertilizer", "fertilisers", "khad", "urvarak"]):
        if not (request.lat and request.lon and request.crop and request.user_id):
            return {
                "error": "Fertilizer recommendation needs user_id, crop name and location (lat/lon).",
                "example": {
                    "query": "Suggest fertilizer for wheat",
                    "lat": 28.6,
                    "lon": 77.2,
                    "crop": "wheat",
                    "user_id": "farmer_123"
                }
            }

        recommendation = get_fertilizer_recommendation(
            user_id=request.user_id,
            crop=request.crop,
            lat=request.lat,
            lon=request.lon
        )
        return {
            "intent": "fertilizer_recommendation",
            "query": request.query,
            "response": recommendation
        }

    # fallback to NLP engine
    result = process_query(
        user_text=request.query,
        user_id=request.user_id,
        lat=request.lat,
        lon=request.lon
    )
    return {
        "intent": "general",
        "query": request.query,
        "response": result
    }
