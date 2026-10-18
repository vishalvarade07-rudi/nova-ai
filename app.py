import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI(title="NOVA AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "app": "NOVA AI",
        "status": "online",
        "message": "Welcome to NOVA AI!"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if not client:
        return {"reply": "NOVA AI setup pending: API key is not configured."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.message,
        )
        return {"reply": response.text or "No response received."}
    except Exception:
        return {
            "reply": "NOVA AI encountered an error. Please try again later."
        }
