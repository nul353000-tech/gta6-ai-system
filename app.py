import os
import time
import base64
import binascii
from collections import defaultdict, deque
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-terra").strip()
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "").strip()
MAX_IMAGE_BYTES = 5 * 1024 * 1024
WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 12
hits = defaultdict(deque)

app = FastAPI(title="GTA 6 AI SYSTEM API", version="1.1.0")
# Android WebView can send requests with an opaque/null origin. No cookies are used;
# the private bearer token is mandatory on the protected endpoint.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000)
    image_data_url: Optional[str] = Field(default=None, max_length=7_000_000)

@app.get("/health")
def health():
    return {"status": "ok", "service": "gta6-ai-system", "version": "1.1.0"}

@app.post("/v1/assistant")
async def assistant(
    payload: AssistantRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    if not APP_ACCESS_TOKEN:
        raise HTTPException(503, "Backend non configuré : APP_ACCESS_TOKEN manquant.")
    if authorization != f"Bearer {APP_ACCESS_TOKEN}":
        raise HTTPException(401, "Accès refusé.")
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY manquante côté serveur.")

    now = time.time()
    ip = request.client.host if request.client else "unknown"
    q = hits[ip]
    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()
    if len(q) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(429, "Trop de requêtes. Réessaie dans une minute.")
    q.append(now)

    content = [{"type": "input_text", "text": payload.prompt}]
    if payload.image_data_url:
        data_url = payload.image_data_url
        if not data_url.startswith("data:image/") or ";base64," not in data_url:
            raise HTTPException(400, "Image invalide : data URL image base64 attendue.")
        header, encoded = data_url.split(",", 1)
        if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 16):
            raise HTTPException(413, "Image trop volumineuse (maximum 5 Mo).")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, "Encodage image invalide.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image trop volumineuse (maximum 5 Mo).")
        mime = header[5:].split(";", 1)[0].lower()
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise HTTPException(400, "Format image non pris en charge.")
        content.append({
            "type": "input_image",
            "image_url": data_url,
            "detail": "auto",
        })

    body = {
        "model": OPENAI_MODEL,
        "store": False,
        "instructions": (
            "Tu es le copilote de GTA 6 AI SYSTEM. Réponds en français, de façon concise, "
            "pratique et structurée. Distingue les faits confirmés, les hypothèses et les "
            "informations inconnues. N'invente pas les mécaniques, missions, cartes ou "
            "statistiques de GTA 6. Si une capture est fournie, décris uniquement ce qui est "
            "visible et signale les incertitudes. Tu ne contrôles pas la PS5 et ne prétends "
            "jamais exécuter des actions dans le jeu."
        ),
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 1000,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            resp = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        if resp.status_code >= 400:
            # Do not relay upstream response bodies; they may contain internal details.
            raise HTTPException(502, f"Erreur du fournisseur IA ({resp.status_code}).")
        data = resp.json()
        texts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
        answer = "\n".join(texts).strip()
        if not answer:
            raise HTTPException(502, "Le fournisseur IA n'a renvoyé aucun texte.")
        return {"answer": answer, "model": OPENAI_MODEL}
    except httpx.TimeoutException:
        raise HTTPException(504, "Délai dépassé pendant la réponse IA.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(502, "Impossible de joindre le service IA.")
