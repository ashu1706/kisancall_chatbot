from fastapi import FastAPI
from pydantic import BaseModel
from app.nlp_engine import process_query

app = FastAPI()

# For testing if server works
@app.get("/")
def root():
    return {"message": "Chatbot backend is running"}

# Data model for chatbot request
class ChatRequest(BaseModel):
    query: str

# Chatbot POST endpoint
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    result = process_query(request.query)
    return result