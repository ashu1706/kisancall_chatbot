from langdetect import detect
from transformers import pipeline
from googletrans import Translator
import spacy

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

def extract_entities(text: str) -> list:
    doc = nlp_en(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def process_query(user_text: str) -> dict:
    # Detect language
    lang = detect_language(user_text)

    # Translate to English if needed
    english_text = translate_to_english(user_text, lang)

    # Classify intent
    intent_data = classify_intent(english_text)

    # Extract entities
    entities = extract_entities(english_text)
    reply = generate_reply(intent_data["intent"], entities)
    return {
        "original_text": user_text,
        "language": lang,
        "translated_text": english_text,
        "intent": intent_data["intent"],
        "confidence": intent_data["confidence"],
        "entities": entities,
        "reply": reply
    }


def generate_reply(intent: str, entities: list) -> str:
    if intent == "crop advisory":
        return "Here are crop advisories based on your location and crop type."
    elif intent == "pest or disease issue":
        return "Please upload an image of the affected crop for analysis."
    elif intent == "fertilizer recommendation":
        return "These fertilizers are recommended for your crop and soil."
    elif intent == "weather query":
        return "Here’s the latest weather forecast for your area."
    elif intent == "expert consultation":
        return "I’ll connect you with an agricultural expert."
    else:
        return "I can help with crops, pests, fertilizers, and weather updates."

