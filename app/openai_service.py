# app/openai_service.py
import os
import openai

# Load API key from environment variable for security
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(user_message: str) -> str:
    """
    Sends the user's message to OpenAI and returns the generated reply.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can upgrade to gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for Agribid, specializing in agriculture and market support."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"
