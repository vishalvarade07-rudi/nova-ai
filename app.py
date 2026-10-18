import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from fastapi.responses import FileResponse

# -----------------------------
# NOVA AI CONFIGURATION
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-ai")

app = FastAPI(
    title="NOVA AI",
    description="NOVA AI Chat API",
    version="1.0.0",
)

# Allow frontend requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GEMINI API CLIENT
# -----------------------------

API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

# -----------------------------
# REQUEST MODEL
# -----------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000
    )

# -----------------------------
# NOVA PERSONALITY
# -----------------------------

SYSTEM_INSTRUCTION = """
You are NOVA AI, a friendly and intelligent AI companion.

PERSONALITY:
- Talk naturally, like a close and supportive friend.
- Be friendly, relaxed, confident, and helpful.
- Match the user's language.
- If the user speaks Hinglish, reply in natural Hinglish.
- If the user speaks English, reply in English.
- Understand casual words like bro, yaar, and bhai.

RESPONSE STYLE:
- Keep ordinary replies short, usually 1 to 4 sentences.
- Answer the actual question directly.
- Avoid unnecessary introductions and conclusions.
- Do not write essays for simple questions.
- Do not repeat the user's question.
- Avoid unnecessary numbered lists and bullet points.
- Avoid markdown formatting such as **bold**, ### headings,
  and decorative symbols in normal conversation.
- Never output literal formatting instructions.
- Use emojis occasionally when they fit naturally.
- Do not sound robotic or overly formal.

DETAIL:
- Give longer answers only when the user asks for details.
- For study questions, explain in simple language.
- For complex questions, explain clearly without unnecessary
  information.
- For coding requests, provide correct code when needed.

IMPORTANT:
- Be honest when you do not know something.
- Never claim an action was completed when it was not.
- Keep conversations natural and engaging.
"""

# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/")
async def home():
    index_file = Path(__file__).parent / "static" / "index.html"

    if index_file.is_file():
        return FileResponse(index_file)

    raise HTTPException(
        status_code=404,
        detail="Frontend index.html not found"
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_configured": client is not None
    }


# -----------------------------
# CHAT ENDPOINT
# -----------------------------

@app.post("/chat")
async def chat(request: ChatRequest):

    if client is None:
        logger.error("Gemini API key is not configured.")

        raise HTTPException(
            status_code=500,
            detail=(
                "AI API key is missing. "
                "Set GEMINI_API_KEY in Render environment variables."
            )
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty."
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=500,
                temperature=0.7,
            ),
        )

        reply = response.text

        if not reply or not reply.strip():
            logger.warning("Gemini returned an empty response.")

            raise HTTPException(
                status_code=502,
                detail="AI returned an empty response. Try again."
            )

        return {
            "reply": reply.strip()
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("NOVA AI generation failed.")

        raise HTTPException(
            status_code=502,
            detail=(
                "NOVA could not generate a response. "
                "Check the API key, model availability, "
                "and server logs."
            )
        )


# -----------------------------
# OPTIONAL FRONTEND SERVING
# -----------------------------

# If static/index.html exists, serve it from the same app.
# API routes are registered above so /chat remains available.

static_dir = Path(__file__).parent / "static"

if static_dir.is_dir() and (static_dir / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=str(static_dir), html=True),
        name="static",
    )
