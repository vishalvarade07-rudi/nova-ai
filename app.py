import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai

app = FastAPI(title="NOVA AI")

app.mount(
"/static",
StaticFiles(directory="static"),
name="static",
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

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
detail="NOVA AI is not configured yet.",
)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=request.message,
    )

    return {
        "reply": response.text or "I couldn't generate a response."
    }

except Exception:
    raise HTTPException(
        status_code=502,
        detail="NOVA AI couldn't get a response. Please try again.",
    )
