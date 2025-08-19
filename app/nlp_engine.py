# app/nlp_engine.py
from langdetect import detect
from transformers import pipeline
from googletrans import Translator
import spacy
from typing import List, Tuple, Optional

from app.location_weather import fetch_weather_for_location, get_weather
from app.user_location import get_user_location
from app.fertilizer_api import get_fertilizer_recommendation
from app.crop_health_api import get_crop_health
from app.pest_api import detect_pests

# Load translation
translator = Translator()

# Load English NER model
nlp_en = spacy.load("en_core_web_sm")

# Load multilingual zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define chatbot intents
INTENTS = [
    "crop advisory",
    "crop health",
    "pest or disease issue",
    "fertilizer recommendation",
    "weather query",
    "expert consultation",
    "general query"
]

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text: str, src_lang: str) -> str:
    if src_lang.lower().startswith("en"):
        return text
    return translator.translate(text, src=src_lang, dest="en").text

def classify_intent(text: str) -> dict:
    result = classifier(text, INTENTS)
    return {
        "intent": result["labels"][0],
        "confidence": result["scores"][0]
    }

def extract_entities(text: str) -> List[Tuple[str, str]]:
    doc = nlp_en(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def _extract_location_from_entities(entities: List[Tuple[str, str]]) -> Optional[str]:
    for ent_text, ent_label in entities:
        if ent_label in ("GPE", "LOC"):
            return ent_text
    return None

def process_query(user_text: str,
                  user_id: Optional[str] = None,
                  lat: Optional[float] = None,
                  lon: Optional[float] = None,
                  crop: Optional[str] = None,
                  image_url: Optional[str] = None) -> dict:
    # Detect language
    lang = detect_language(user_text)

    # Translate to English if needed
    english_text = translate_to_english(user_text, lang)

    # Classify intent
    intent_data = classify_intent(english_text)

    # Extract entities
    entities = extract_entities(english_text)

    # Generate reply
    reply = generate_reply(
        intent=intent_data["intent"],
        entities=entities,
        user_id=user_id,
        lat=lat,
        lon=lon,
        crop=crop,
        image_url=image_url,
        original_text=user_text
    )

    return {
        "original_text": user_text,
        "language": lang,
        "translated_text": english_text,
        "intent": intent_data["intent"],
        "confidence": intent_data["confidence"],
        "entities": entities,
        "reply": reply
    }

def generate_reply(intent: str,
                   entities: List[Tuple[str, str]],
                   user_id: Optional[str] = None,
                   lat: Optional[float] = None,
                   lon: Optional[float] = None,
                   crop: Optional[str] = None,
                   image_url: Optional[str] = None,
                   original_text: Optional[str] = None) -> str:
    """
    Routes queries to appropriate modules depending on intent
    """
    # ---------- Weather ----------
    if intent == "weather query":
        if lat is not None and lon is not None:
            w = get_weather(lat, lon)
            if "error" in w:
                return "Sorry, I couldn’t fetch the weather right now."
            return f"Current weather: {w['temperature']}°C, wind {w['windspeed']} km/h."

        if user_id:
            stored = get_user_location(user_id)
            if stored and "lat" in stored and "lon" in stored:
                w = get_weather(stored["lat"], stored["lon"])
                if "error" in w:
                    return "Sorry, I couldn’t fetch the weather right now."
                loc_name = stored.get("location_name") or "your saved location"
                return f"Weather for {loc_name}: {w['temperature']}°C, wind {w['windspeed']} km/h."

        place = _extract_location_from_entities(entities)
        if place:
            data = fetch_weather_for_location(place)
            if "error" in data or "weather" not in data:
                return f"Sorry, I couldn’t find weather for {place}."
            w = data["weather"]
            return f"Weather in {place}: {w['temperature']}°C, wind {w['windspeed']} km/h."

        return "Please share your location (lat/lon or place name) to get weather updates."

    # ---------- Fertilizer ----------
    if intent == "fertilizer recommendation":
        if not crop:
            return "Please specify your crop so I can suggest fertilizers."
        if lat is not None and lon is not None:
            return get_fertilizer_recommendation(crop, lat, lon)
        return f"Fertilizer recommendation: I need your location to provide accurate advice for {crop}."

    # ---------- Crop Health ----------
    if intent == "crop health":
        if lat is not None and lon is not None:
            return get_crop_health(lat, lon, crop)
        return "Please provide your farm location (lat/lon) to analyze crop health."

    # ---------- Pest / Disease ----------
    if intent == "pest or disease issue":
        if not image_url:
            return "Please upload an image of the affected crop for analysis."
        return detect_pests(image_url, crop)

    # ---------- Crop Advisory (weather + general rules) ----------
    if intent == "crop advisory":
        if lat is not None and lon is not None:
            w = get_weather(lat, lon)
            if "error" in w:
                return "Crop advisory: I couldn’t fetch weather for your area right now."
            temp = w.get("temperature")
            wind = w.get("windspeed")
            return (
                "Crop Advisory:\n"
                f"- Current temperature: {temp}°C; wind {wind} km/h.\n"
                "- Avoid irrigation during peak heat (>35°C).\n"
                "- Delay fertilizer if heavy rain is expected in the next 48h."
            )
        return "Crop advisory: please share your location to tailor recommendations."

    # ---------- Expert ----------
    if intent == "expert consultation":
        return "I’ll connect you with an agricultural expert."

    return "I can help with crops, pests, fertilizers, crop health, and weather updates."
