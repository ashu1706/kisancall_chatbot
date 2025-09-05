import os
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import json

# Get absolute BASE_DIR (your project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to model + labels
MODEL_PATH = os.path.join(BASE_DIR, "model_train", "xlm_intent_model")
LABEL_PATH = os.path.join(BASE_DIR, "model_train", "label_mapping.json")

# Load model + tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# Load label mapping
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label2id = json.load(f)

# Invert mapping for prediction
id2label = {int(v): k for k, v in label2id.items()}

def predict_intent(query: str):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
    return id2label[pred_id], probs[0][pred_id].item()

# Quick test
if __name__ == "__main__":
    test_queries = [
        "What is the weather in Delhi today?",
        "Suggest fertilizer for my wheat crop",
        "Are there any government schemes for farmers?",
        "What price is onion selling in the market?",
        "I have pest issues in my rice field"
    ]
    for q in test_queries:
        intent, conf = predict_intent(q)
        print(f"Query: {q}")
        print(f"Predicted intent: {intent} (confidence: {conf:.4f})\n")
