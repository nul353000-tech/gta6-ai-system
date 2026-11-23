import os
import time
import base64
import binascii
import hmac
from collections import defaultdict, deque
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-terra").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").strip().lower()
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "").strip()
MAX_IMAGE_BYTES = 5 * 1024 * 1024
WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 12
MAX_RATE_LIMIT_CLIENTS = 10_000
hits = defaultdict(deque)

app = FastAPI(title="GTA 6 AI SYSTEM API", version="1.2.0")
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")
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


def _parse_image(data_url: str) -> tuple[str, str, str]:
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
    return data_url, declared_mime, encoded

@app.post("/v1/assistant")
async def assistant(
    payload: AssistantRequest,
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
):
    if not APP_ACCESS_TOKEN:
        raise HTTPException(503, "Backend non configuré : APP_ACCESS_TOKEN manquant.")
    if (
        not authorization
        or authorization.scheme.lower() != "bearer"
        or not hmac.compare_digest(authorization.credentials, APP_ACCESS_TOKEN)
    ):
        raise HTTPException(401, "Accès refusé.")
    provider = AI_PROVIDER
    if provider == "auto":
        if GEMINI_API_KEY:
            provider = "gemini"
        elif OPENAI_API_KEY:
            provider = "openai"
        else:
            raise HTTPException(
                503,
                "Aucune clé IA configurée. Ajoute GEMINI_API_KEY pour le niveau gratuit Gemini.",
            )
    if provider not in {"openai", "gemini"}:
        raise HTTPException(503, "AI_PROVIDER doit être 'auto', 'openai' ou 'gemini'.")
    if provider == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY manquante côté serveur.")
    if provider == "openai" and not OPENAI_API_KEY:
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

    data_url = image_mime = image_encoded = None
    if payload.image_data_url:
        data_url, image_mime, image_encoded = _parse_image(payload.image_data_url)

    instructions = (
        "Tu es le copilote de GTA 6 AI SYSTEM. Réponds en français, de façon concise, "
        "pratique et structurée. Distingue les faits confirmés, les hypothèses et les "
        "informations inconnues. N'invente pas les mécaniques, missions, cartes ou "
        "statistiques de GTA 6. Si une capture est fournie, décris uniquement ce qui est "
        "visible et signale les incertitudes. Tu ne contrôles pas la PS5 et ne prétends "
        "jamais exécuter des actions dans le jeu."
    )

    if provider == "gemini":
        parts = [{"text": f"{instructions}\n\n{payload.prompt}"}]
        if image_encoded and image_mime:
            parts.append({"inline_data": {"mime_type": image_mime, "data": image_encoded}})
        request_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        request_headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
        request_body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 1000},
        }
        model = GEMINI_MODEL
    else:
        content = [{"type": "input_text", "text": payload.prompt}]
        if data_url:
            content.append({
                "type": "input_image",
                "image_url": data_url,
                "detail": "auto",
            })
        request_url = "https://api.openai.com/v1/responses"
        request_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        request_body = {
            "model": OPENAI_MODEL,
            "store": False,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 1000,
        }
        model = OPENAI_MODEL
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            resp = await client.post(request_url, headers=request_headers, json=request_body)
        if resp.status_code >= 400:
            # Do not relay upstream response bodies; they may contain internal details.
            if provider == "gemini" and resp.status_code in {400, 401, 403, 404}:
                try:
                    error = resp.json().get("error", {})
                    error = error if isinstance(error, dict) else {}
                    message = error.get("message", "")
                    message = message.lower() if isinstance(message, str) else ""
                    status = error.get("status", "")
                    status = status.upper() if isinstance(status, str) else ""
                except (ValueError, AttributeError):
                    message = status = ""
                if "api key not valid" in message or "api_key_invalid" in message:
                    raise HTTPException(
                        502,
                        "Gemini rejette GEMINI_API_KEY. Vérifie la clé Google AI Studio dans Render.",
                    )
                if "generative language api" in message and any(
                    phrase in message for phrase in ("not been used", "disabled", "not enabled")
                ):
                    raise HTTPException(
                        502,
                        "L'API Gemini n'est pas activée pour le projet de cette clé Google.",
                    )
                if "model" in message and any(
                    phrase in message for phrase in ("not found", "not supported", "unknown")
                ):
                    raise HTTPException(
                        502,
                        f"Modèle Gemini indisponible ({GEMINI_MODEL}). Vérifie GEMINI_MODEL dans Render.",
                    )
                if status and len(status) <= 64 and status.replace("_", "").isalnum():
                    raise HTTPException(
                        502,
                        f"Gemini a rejeté la requête ({resp.status_code}, {status}). Vérifie la clé et le modèle configurés dans Render.",
                    )
                raise HTTPException(
                    502,
                    f"Gemini a rejeté la requête ({resp.status_code}). Vérifie GEMINI_API_KEY et GEMINI_MODEL dans Render.",
                )
            if resp.status_code == 429:
                if provider == "gemini":
                    raise HTTPException(
                        429,
                        "Quota gratuit ou limite de débit Gemini atteint. Vérifie les quotas du projet Google AI Studio.",
                    )
                try:
                    error = resp.json().get("error", {})
                    error = error if isinstance(error, dict) else {}
                except (ValueError, AttributeError):
                    error = {}
                code = error.get("code")
                error_type = error.get("type")
                quota_codes = {
                    "insufficient_quota",
                    "usage_limit_exceeded",
                    "organization_usage_limit_exceeded",
                    "project_spend_limit_exceeded",
                    "organization_spend_limit_exceeded",
                }
                if (isinstance(code, str) and code in quota_codes) or error_type == "insufficient_quota":
                    raise HTTPException(
                        429,
                        "Quota ou plafond de dépenses OpenAI atteint. Vérifie la facturation et les limites du projet.",
                    )

                detail = "Limite de débit OpenAI atteinte. Réduis la fréquence des requêtes et réessaie."
                if isinstance(code, str) and len(code) <= 64 and code.replace("_", "").replace("-", "").isalnum():
                    detail += f" (code : {code})"
                retry_after = resp.headers.get("Retry-After", "")
                headers = {"Retry-After": retry_after} if retry_after.isdigit() else None
                raise HTTPException(429, detail, headers=headers)
            raise HTTPException(502, f"Erreur du fournisseur IA ({resp.status_code}).")
        data = resp.json()
        texts = []
        if provider == "gemini":
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if isinstance(part.get("text"), str):
                        texts.append(part["text"])
        else:
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            texts.append(part.get("text", ""))
        answer = "\n".join(texts).strip()
        if not answer:
            raise HTTPException(502, "Le fournisseur IA n'a renvoyé aucun texte.")
        return {"answer": answer, "model": model}
    except httpx.TimeoutException:
        raise HTTPException(504, "Délai dépassé pendant la réponse IA.")
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(502, "Impossible de joindre le service IA.")

