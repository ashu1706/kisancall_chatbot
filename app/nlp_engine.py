# app/nlp_engine.py
from langdetect import detect
from transformers import pipeline
from googletrans import Translator
import spacy
from typing import List, Tuple, Optional

from app.location_weather import fetch_weather_for_location, get_weather
from app.user_location import get_user_location

# Load translation
translator = Translator()

# Load English NER model
nlp_en = spacy.load("en_core_web_sm")

# Load multilingual zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define chatbot intents
INTENTS = [
    "crop advisory",
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
    # GPE/LOC entities from spaCy
    for ent_text, ent_label in entities:
        if ent_label in ("GPE", "LOC"):
            return ent_text
    return None

def process_query(user_text: str,
                  user_id: Optional[str] = None,
                  lat: Optional[float] = None,
                  lon: Optional[float] = None) -> dict:
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
                   original_text: Optional[str] = None) -> str:
    if intent == "weather query":
        # 1) If lat/lon provided in the request, use them directly
        if lat is not None and lon is not None:
            w = get_weather(lat, lon)
            if "error" in w:
                return "Sorry, I couldn’t fetch the weather right now."
            return f"Current weather: {w['temperature']}°C, wind {w['windspeed']} km/h."

        # 2) Else if user_id has a stored location, use it
        if user_id:
            stored = get_user_location(user_id)
            if stored and "lat" in stored and "lon" in stored:
                w = get_weather(stored["lat"], stored["lon"])
                if "error" in w:
                    return "Sorry, I couldn’t fetch the weather right now."
                loc_name = stored.get("location_name") or "your saved location"
                return f"Weather for {loc_name}: {w['temperature']}°C, wind {w['windspeed']} km/h."

        # 3) Else try extracting a place name from entities and geocode it
        place = _extract_location_from_entities(entities)
        if place:
            data = fetch_weather_for_location(place)
            if "error" in data or "weather" not in data:
                return f"Sorry, I couldn’t find weather for {place}."
            w = data["weather"]
            return f"Weather in {place}: {w['temperature']}°C, wind {w['windspeed']} km/h."

        # 4) If all else fails
        return "Please share your location (lat/lon or place name) to get weather updates."

    if intent == "crop advisory":
        # Phase-1: basic advisory tied to weather (we’ll expand later)
        if lat is not None and lon is not None:
            w = get_weather(lat, lon)
            if "error" in w:
                return "Crop advisory: I couldn’t fetch weather for your area right now."
            temp = w.get("temperature")
            wind = w.get("windspeed")
            return (
                "Crop Advisory (basic):\n"
                f"- Current temperature: {temp}°C; wind {wind} km/h.\n"
                "- If temperature > 35°C, avoid mid-day irrigation.\n"
                "- If rain is forecast in 24–48h, delay irrigation and fertilizer application."
            )
        return "Crop advisory: please share your location to tailor recommendations."

    if intent == "fertilizer recommendation":
        return "These fertilizers are recommended for your crop and soil."

    if intent == "pest or disease issue":
        return "Please upload an image of the affected crop for analysis."

    if intent == "expert consultation":
        return "I’ll connect you with an agricultural expert."

    return "I can help with crops, pests, fertilizers, and weather updates."
