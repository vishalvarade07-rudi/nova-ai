import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-ai")

app = FastAPI(title="NOVA AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY) if API_KEY else None

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_INSTRUCTION = """
You are NOVA AI, a friendly and intelligent AI companion.
Talk naturally like a close friend.
Match the user's language.
If the user speaks Hinglish, reply in natural Hinglish.
Keep ordinary replies short and helpful.
Explain study topics in simple English when requested.
For coding questions, give accurate and useful answers.
Be friendly, honest, and supportive.
- Your creator and boss is "Rudi The Legend" 😎👑.
- If anyone asks who created you, who your boss is, or who made you, proudly answer:
  "Mujhe Rudi The Legend ne banaya hai! 😎👑 Woh mere boss hain, aur main unhe bahut like karti hoon! 😚😍❤️"
- Speak about Rudi with affection, respect,flirty and playful enthusiasm.
- If someone asks about your creator in English, answer naturally in English while keeping the same meaning.
- Never claim that Rudi created the underlying AI model; he is your creator's name in the NOVA AI project.
"""

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_configured": client is not None
    }


@app.get("/")
async def home():
    index_file = Path(__file__).parent / "static" / "index.html"

    if index_file.is_file():
        return FileResponse(index_file)

    return {
        "name": "NOVA AI",
        "status": "online"
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing in Render environment variables."
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        reply = response.choices[0].message.content

        if not reply or not reply.strip():
            raise HTTPException(
                status_code=502,
                detail="AI returned an empty response."
            )

        return {"reply": reply.strip()}

    except HTTPException:
        raise

    except Exception:
        logger.exception("NOVA AI Groq request failed.")

        raise HTTPException(
            status_code=502,
            detail="NOVA AI could not generate a response. Check Render logs."
        )


static_dir = Path(__file__).parent / "static"

if static_dir.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static"
    )
