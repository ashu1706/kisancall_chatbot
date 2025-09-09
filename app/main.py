# app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import shutil, os, logging, asyncio
import joblib
import pandas as pd
from app.database import database
from app.user_location import set_user_location, get_user_location
from app.location_weather import fetch_weather_for_location, get_weather
from app.user_polygons import set_polygon, get_polygon

# New imports for PostgreSQL agricultural chatbot
from app.agri_chatbot import get_chatbot
from app.advisory_utils import (
    prepare_irrigation_input,
    prepare_fertilizer_input,
    prepare_pest_input
)
from app.intent import predict_intent

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Kisan Call Chatbot API - PostgreSQL Enhanced")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot initialization flag and model storage
chatbot_initialized = False
initialization_error = None
ml_models = {}


# --- Helper functions to load ML models and make predictions ---
def get_advisory_from_models(crop: str, weather_data: dict, user_context: dict) -> dict:
    try:
        # Prepare inputs
        irrigation_input = prepare_irrigation_input({'weather_data': weather_data, 'user_context': user_context, 'soil_data': user_context.get('soil_data', {})})
        fertilizer_input = prepare_fertilizer_input({'weather_data': weather_data, 'user_context': user_context, 'soil_data': user_context.get('soil_data', {})})
        pest_input = prepare_pest_input({'weather_data': weather_data, 'user_context': user_context})

        # Predictions
        irrigation = ml_models['irrigation'].predict(ml_models['preprocessor_irrigation'].transform(irrigation_input))[0]
        fert_type = ml_models['fertilizer_type'].predict(ml_models['preprocessor_fertilizer'].transform(fertilizer_input))[0]
        fert_qty = ml_models['fertilizer_quantity'].predict(ml_models['preprocessor_fertilizer'].transform(fertilizer_input))[0]
        pest = ml_models['pest'].predict(ml_models['preprocessor_pest'].transform(pest_input))[0]

        return {
            "irrigation": f"{irrigation:.2f} mm",
            "fertilizer": {
                "type": fert_type,
                "quantity": f"{fert_qty:.2f} kg/ha"
            },
            "pest_risk": str(pest)
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------- Startup / Shutdown -----------------
@app.on_event("startup")
async def startup():
    global chatbot_initialized, initialization_error, ml_models

    # Connect to database
    await database.connect()

    async def initialize_systems():
        global chatbot_initialized, initialization_error, ml_models
        try:
            logger.info("Initializing all systems...")

            # 1. Initialize Agricultural Chatbot (from your existing code)
            chatbot = await get_chatbot()

            # 2. Load ML models from the 'models' directory
            MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
            ml_models['irrigation'] = joblib.load(os.path.join(MODELS_DIR, 'irrigation_model.joblib'))
            ml_models['preprocessor_irrigation'] = joblib.load(
                os.path.join(MODELS_DIR, 'data_preprocessor_irrigation.joblib'))
            ml_models['fertilizer_type'] = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_type_model.joblib'))
            ml_models['fertilizer_quantity'] = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_quantity_model.joblib'))
            ml_models['preprocessor_fertilizer'] = joblib.load(
                os.path.join(MODELS_DIR, 'data_preprocessor_fertilizer.joblib'))
            ml_models['pest'] = joblib.load(os.path.join(MODELS_DIR, 'pest_advisory_model.joblib'))
            ml_models['preprocessor_pest'] = joblib.load(os.path.join(MODELS_DIR, 'data_preprocessor_pest.joblib'))

            chatbot_initialized = True
            logger.info("✅ All systems initialized successfully!")
        except Exception as e:
            initialization_error = str(e)
            logger.error(f"❌ Failed to initialize systems: {e}")
            logger.info("Chatbot will use fallback mode")

    asyncio.create_task(initialize_systems())


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# ----------------- Enhanced Request Models -----------------
class ChatRequest(BaseModel):
    query: str = Field(..., example="What is the weather in Delhi today?")
    user_id: Optional[str] = Field(None, example="farmer_123")
    lat: Optional[float] = Field(None, example=28.6139)
    lon: Optional[float] = Field(None, example=77.2090)
    crop: Optional[str] = Field(None, example="wheat")
    language: Optional[str] = Field("en", example="en")
    context: Optional[Dict[str, Any]] = Field(None, example={"season": "kharif"})


class ChatResponse(BaseModel):
    query: str
    response: str
    confidence: float
    source: str
    intent: Optional[str] = None
    suggestions: List[str] = []
    weather_context: Optional[Dict] = None
    location_context: Optional[Dict] = None
    advisory: Optional[Dict] = None
    metadata: Optional[Dict] = None


class LocationUpdate(BaseModel):
    user_id: str
    lat: float
    lon: float
    location_name: Optional[str] = None


class PolygonRequest(BaseModel):
    user_id: str
    polygon_id: str


# ----------------- Upload Routes (Existing) -----------------
@app.post("/upload/{file_type}")
async def upload_file(user_id: str, file: UploadFile = File(...), file_type: str = "image"):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query = """
        INSERT INTO user_uploads (user_id, file_name, file_type, file_path)
        VALUES (:user_id, :file_name, :file_type, :file_path)
    """
    values = {
        "user_id": user_id,
        "file_name": file.filename,
        "file_type": file_type,
        "file_path": file_path
    }
    await database.execute(query=query, values=values)
    return {"status": "success", "filename": file.filename, "path": file_path}


# ----------------- Location & Weather Routes (Existing) -----------------
@app.get("/")
def root():
    return {
        "message": "Enhanced Kisan Call Chatbot backend is running",
        "version": "2.0.0-postgresql",
        "chatbot_initialized": chatbot_initialized
    }


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
    polygon_id = set_polygon(req.user_id, req.polygon_id)
    return {"message": "Polygon linked successfully", "user_id": req.user_id, "polygon_id": polygon_id}

# ----------------- Core Chat Endpoint -----------------
@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    if not chatbot_initialized:
        if initialization_error:
            logger.warning(f"Chatbot not initialized due to error: {initialization_error}")
        return await fallback_chat_endpoint(request)

    chatbot = await get_chatbot()

    location_context = None
    weather_context = None
    if request.lat and request.lon:
        location_context = {"lat": request.lat, "lon": request.lon}
        weather_info = get_weather(request.lat, request.lon)
        if weather_info:
            weather_context = weather_info

    elif request.user_id:
        stored_location = get_user_location(request.user_id)
        if stored_location:
            location_context = stored_location
            if stored_location.get("lat") and stored_location.get("lon"):
                weather_info = get_weather(stored_location["lat"], stored_location["lon"])
                if weather_info:
                    weather_context = weather_info
            elif stored_location.get("location_name"):
                weather_info = fetch_weather_for_location(stored_location["location_name"])
                if weather_info:
                    weather_context = weather_info.get("weather", {})

    advisory_data = None
    if request.crop and weather_context and "current" in weather_context:
        if any(word in request.query.lower() for word in ["advisory", "recommendation", "advice", "fertilizer", "what to do"]):
            user_context = {
                "crop": request.crop,
                "location_name": location_context.get("location_name", "Unknown") if location_context else "Unknown",
                "sowing_season": "Rabi",
                "soil_type": "Loamy",
                "growth_stage": "Vegetative"
            }
            advisory_data = get_advisory_from_models(request.crop, weather_context, user_context)

    try:
        enhanced_query = request.query
        context_info = {}

        if request.crop:
            context_info['crop'] = request.crop
            enhanced_query += f" (crop: {request.crop})"
        if location_context:
            context_info['location'] = location_context['location_name']
            enhanced_query += f" (location: {location_context['location_name']})"
        if weather_context and "current" in weather_context:
            current_weather = weather_context["current"]
            context_info['weather'] = f"Temperature: {current_weather.get('temperature', 'N/A')}°C"

        result = await chatbot.process_query(enhanced_query, {
            "crop": request.crop,
            "location": location_context,
            "weather": weather_context,
            "advisory_data": advisory_data
        })

        enhanced_response = result['response']
        if weather_context and any(word in request.query.lower() for word in ["weather", "rain", "temperature"]):
            enhanced_response = enhance_response_with_weather(enhanced_response, weather_context)

        # ✅ Use your XLM intent model
        intent, _ = predict_intent(request.query)

        suggestions = result.get("suggestions", [])
        if result["confidence"] < 0.75:
            if request.crop:
                suggestions.append(f"Ask more specific questions about {request.crop}")
            if not location_context:
                suggestions.append("Share your location for weather-specific advice")
            suggestions.append("Upload images for visual diagnosis")

        background_tasks.add_task(
            log_query_analytics,
            request.user_id,
            request.query,
            result["query_type"],
            result["confidence"],
            result["source"],
            result["processing_time"],
            context_info
        )

        return ChatResponse(
            query=request.query,
            response=enhanced_response,
            confidence=result["confidence"],
            source=result["source"],
            intent=intent,
            suggestions=suggestions,
            weather_context=weather_context,
            location_context=location_context,
            advisory=advisory_data,
            metadata={
                "context_info": context_info,
                "processing_time": result["processing_time"],
                "language": request.language,
                "query_type": result["query_type"]
            }
        )

    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        return await fallback_chat_endpoint(request)


# ----------------- Fallback Endpoint -----------------
async def fallback_chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        intent, confidence = predict_intent(request.query)
    except:
        intent, confidence = "general_query", 0.5

    replies = {
        "fertilizer_management": f"For {request.crop or 'your crop'}, use NPK fertilizers in recommended ratios.",
        "seed_recommendation": f"Use certified seeds for {request.crop or 'your crop'}.",
        "pest_disease_issue": "Please upload images of affected plants for diagnosis.",
        "weather_advisory": "Checking weather conditions for your location.",
        "government_scheme": "Schemes: PM-Kisan, Crop Insurance, KCC loans.",
        "market_info": f"Fetching current market prices for {request.crop or 'crops'}."
    }

    reply = replies.get(intent, "I'm here to help with farming questions. Ask about crops, weather, pests, fertilizers, or schemes.")

    return ChatResponse(
        query=request.query,
        response=reply,
        confidence=confidence,
        source="xlm_intent_model",
        intent=intent,
        suggestions=["Be more specific about your farming issue", "Upload images for better diagnosis"],
        weather_context=None,
        location_context=None,
        metadata={'fallback_mode': True}
    )


# ----------------- Helper -----------------
def enhance_response_with_weather(response: str, weather_context: Dict) -> str:
    if not weather_context or 'current' not in weather_context:
        return response

    current = weather_context['current']
    weather_summary = []
    if 'temperature' in current:
        weather_summary.append(f"Current temperature: {current['temperature']}°C")
    if 'windspeed' in current:
        weather_summary.append(f"Wind: {current['windspeed']} km/h")
    if weather_context.get('recent', {}).get('rainfall'):
        recent_rain = sum(weather_context['recent']['rainfall'][-24:])
        if recent_rain > 0:
            weather_summary.append(f"Recent rainfall: {recent_rain:.1f}mm")

    if weather_summary:
        return f"{response}\n\n🌤️ **Weather Context**: {' | '.join(weather_summary)}"
    return response

# ----------------- PostgreSQL Agricultural Endpoints -----------------
@app.get("/chatbot/health")
async def chatbot_health():
    """Check PostgreSQL chatbot system health"""

    db_connected = database.is_connected if hasattr(database, 'is_connected') else True

    # Test vector extension if chatbot is initialized
    vector_extension_available = False
    if chatbot_initialized:
        try:
            result = await database.fetch_one("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            vector_extension_available = result is not None
        except:
            pass

    return {
        "chatbot_initialized": chatbot_initialized,
        "initialization_error": initialization_error,
        "database_connected": db_connected,
        "vector_extension_available": vector_extension_available,
        "version": "2.0.0-postgresql",
        "features": [
            "postgresql_vector_storage",
            "confidence_scoring",
            "weather_integration",
            "location_context",
            "multilingual_support",
            "intent_classification",
            "query_analytics"
        ]
    }


@app.get("/chatbot/stats")
async def chatbot_stats():
    """Get PostgreSQL chatbot statistics"""

    if not chatbot_initialized:
        return {"error": "Chatbot not initialized", "initialization_error": initialization_error}

    try:
        chatbot = await get_chatbot()
        stats = await chatbot.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": str(e)}


# Define a Pydantic model for the feedback request body
class FeedbackRequest(BaseModel):
    user_id: str
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None


@app.get("/chatbot/analytics")
async def chatbot_analytics(days: int = 7):
    """Get chatbot analytics for the last N days"""

    try:
        analytics_query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total_queries,
            AVG(confidence_score) as avg_confidence,
            COUNT(CASE WHEN confidence_score >= 0.75 THEN 1 END) as high_confidence_queries,
            COUNT(CASE WHEN confidence_score >= 0.55 AND confidence_score < 0.75 THEN 1 END) as medium_confidence_queries,
            COUNT(CASE WHEN confidence_score < 0.55 THEN 1 END) as low_confidence_queries,
            COUNT(DISTINCT user_id) as unique_users,
            AVG(processing_time_ms) as avg_processing_time_ms
        FROM query_analytics 
        WHERE timestamp >= NOW() - INTERVAL %s DAY
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """

        results = await database.fetch_all(analytics_query, [days])

        analytics = []
        for row in results:
            analytics.append({
                "date": str(row['date']),
                "total_queries": row['total_queries'],
                "avg_confidence": round(float(row['avg_confidence']) if row['avg_confidence'] else 0, 3),
                "high_confidence_queries": row['high_confidence_queries'],
                "medium_confidence_queries": row['medium_confidence_queries'],
                "low_confidence_queries": row['low_confidence_queries'],
                "unique_users": row['unique_users'],
                "avg_processing_time_ms": round(
                    float(row['avg_processing_time_ms']) if row['avg_processing_time_ms'] else 0, 2)
            })

        return {"analytics": analytics, "period_days": days}

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {"error": str(e)}


# ----------------- Quick Test Endpoint -----------------
@app.post("/test/chat")
async def test_chat_quickly(query: str = "How to control cotton pests?", user_id: str = "test_user"):
    """Quick test endpoint for development"""
    request = ChatRequest(
        query=query,
        user_id=user_id,
        crop="cotton",
        language="en"
    )
    return await enhanced_chat_endpoint(request, BackgroundTasks())

# Define a Pydantic model for the feedback request body
class FeedbackRequest(BaseModel):
    user_id: str
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)