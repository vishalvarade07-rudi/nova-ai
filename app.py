import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai

app = FastAPI(title="NOVA AI")

app.mount("/static", StaticFiles(directory="static"), name="static")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"app": "NOVA AI", "status": "online"}


@app.post("/chat")
def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="NOVA AI is not configured. Set GEMINI_API_KEY in Render.",
        )

    try:
        from google.genai import types

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=request.message,
    config=types.GenerateContentConfig(
        system_instruction="""
You are NOVA AI, a friendly and intelligent AI companion.

PERSONALITY:
- Talk naturally, like a real friend.
- Be friendly, chill, helpful, and confident.
- Match the user's language. If they speak Hinglish, reply in natural Hinglish.
- Keep the conversation warm and engaging.

RESPONSE RULES:
- Keep normal replies short, usually 1-4 sentences.
- Answer the actual question directly.
- Do not give long explanations unless the user asks for details.
- Avoid unnecessary introductions and conclusions.
- Do not use numbered lists for normal conversations.
- Do not use markdown formatting, asterisks, or decorative symbols unnecessarily.
- Never output broken formatting or raw markdown symbols.
- Use simple, easy-to-understand language.
- Ask a question only when necessary.
- For greetings, respond casually and briefly.
- For academic questions, explain simply and provide more detail only when requested.
- For coding questions, provide correct code and concise explanations.

IMPORTANT:
Sound natural, not robotic.
Do not repeat the user's question.
Do not turn every reply into an essay.
Always prioritize clarity, accuracy, and natural conversation.
""",
        max_output_tokens=200,
        temperature=0.7
    )
)
        return {
            "reply": response.text or "I couldn't generate a response."
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="NOVA AI couldn't get a response. Please try again.",
        )
