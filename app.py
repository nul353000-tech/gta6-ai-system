import os
import time
import base64
import binascii
import hmac
from collections import defaultdict, deque
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-terra").strip()
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "").strip()
MAX_IMAGE_BYTES = 5 * 1024 * 1024
WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 12
MAX_RATE_LIMIT_CLIENTS = 10_000
hits = defaultdict(deque)

app = FastAPI(title="GTA 6 AI SYSTEM API", version="1.1.2")
authorization_header = APIKeyHeader(name="Authorization", auto_error=False, scheme_name="BearerAuth")
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
    return {"status": "ok", "service": "gta6-ai-system", "version": app.version}


def _image_mime(data: bytes) -> Optional[str]:
    """Return the image MIME type from its signature, not its user supplied name."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _parse_image(data_url: str) -> str:
    if not data_url.startswith("data:image/") or ";base64," not in data_url:
        raise HTTPException(400, "Image invalide : data URL image base64 attendue.")
    header, encoded = data_url.split(",", 1)
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
        raise HTTPException(413, "Image trop volumineuse (maximum 5 Mo).")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "Encodage image invalide.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image trop volumineuse (maximum 5 Mo).")
    declared_mime = header[5:].split(";", 1)[0].lower()
    actual_mime = _image_mime(raw)
    if actual_mime is None or declared_mime != actual_mime:
        raise HTTPException(400, "Le contenu de l'image ne correspond pas à son format déclaré.")
    return data_url

@app.post("/v1/assistant")
async def assistant(
    payload: AssistantRequest,
    request: Request,
    authorization: Optional[str] = Security(authorization_header),
):
    if not APP_ACCESS_TOKEN:
        raise HTTPException(503, "Backend non configuré : APP_ACCESS_TOKEN manquant.")
    supplied_token = authorization.removeprefix("Bearer ") if authorization else ""
    if not hmac.compare_digest(supplied_token, APP_ACCESS_TOKEN) or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Accès refusé.")
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY manquante côté serveur.")

    now = time.time()
    ip = request.client.host if request.client else "unknown"
    if ip not in hits and len(hits) >= MAX_RATE_LIMIT_CLIENTS:
        # Expire idle clients before admitting a new key so attacker-controlled
        # client addresses cannot grow this process-local cache without bound.
        for client_ip, events in list(hits.items()):
            if not events or now - events[-1] > WINDOW_SECONDS:
                del hits[client_ip]
        if len(hits) >= MAX_RATE_LIMIT_CLIENTS:
            hits.pop(next(iter(hits)))
    q = hits[ip]
    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()
    if len(q) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(429, "Trop de requêtes. Réessaie dans une minute.")
    q.append(now)

    content = [{"type": "input_text", "text": payload.prompt}]
    if payload.image_data_url:
        data_url = _parse_image(payload.image_data_url)
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
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(502, "Impossible de joindre le service IA.")

