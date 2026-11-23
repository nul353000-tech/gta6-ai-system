import base64

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

import app as api


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(api, "APP_ACCESS_TOKEN", "test-secret-token")
    monkeypatch.setattr(api, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(api, "OPENAI_MODEL", "gpt-6.1-terra")
    monkeypatch.setattr(api, "GEMINI_API_KEY", "")
    monkeypatch.setattr(api, "GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(api, "AI_PROVIDER", "auto")
    api.hits.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_reports_running_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "gta6-ai-system",
        "version": api.app.version,
    }


def test_dashboard_serves_v13_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "GTA 6 AI SYSTEM V1.3" in response.text
    assert "askAssistant()" in response.text


def test_docker_image_includes_the_dashboard():
    with open("Dockerfile", encoding="utf-8") as dockerfile:
        assert "COPY web/ ./web/" in dockerfile.read()


def test_state_api_requires_bearer_token(client):
    assert client.get("/api/state").status_code == 401
    assert client.post("/api/state", json={"notes": "test"}).status_code == 401


def test_state_api_saves_and_loads_json(client, monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(api, "STATE_FILE", state_file)
    headers = {"Authorization": "Bearer test-secret-token"}
    state = {"notes": "Local save", "journal": [{"text": "test"}]}

    saved = client.post("/api/state", headers=headers, json=state)
    loaded = client.get("/api/state", headers=headers)

    assert saved.status_code == 200
    assert saved.json() == {"saved": True}
    assert loaded.status_code == 200
    assert loaded.json() == {"state": state, "initialized": True}
    assert state_file.exists()


def test_empty_state_reports_not_initialized(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "STATE_FILE", tmp_path / "missing-state.json")
    response = client.get("/api/state", headers={"Authorization": "Bearer test-secret-token"})
    assert response.status_code == 200
    assert response.json()["initialized"] is False


def test_state_api_rejects_invalid_or_oversized_json(client):
    headers = {"Authorization": "Bearer test-secret-token"}
    invalid_type = client.post("/api/state", headers=headers, content="{}")
    invalid_json = client.post(
        "/api/state", headers={**headers, "Content-Type": "application/json"}, content="not-json"
    )
    oversized = client.post(
        "/api/state",
        headers={**headers, "Content-Type": "application/json"},
        content=b"{" + b" " * api.MAX_STATE_BYTES + b"}",
    )

    assert invalid_type.status_code == 415
    assert invalid_json.status_code == 400
    assert oversized.status_code == 413


def test_openapi_documents_bearer_authentication(client):
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["BearerAuth"] == {
        "type": "http", "scheme": "bearer"
    }


def test_m:ÖÚ$z{-®éÜj×Üİ
ˆ‹İŒKØ\ÜÚ\İ[‹ˆXY\œÏ^È]]Üš^˜][Ûˆˆ™X\™\ˆ\İ\ÙXÜ™]]ÚÙ[ˆŸKˆœÛÛ^Èœ›Û\ˆ›Ûš›İ\ˆŸKˆ
Bˆ\ÜÙ\™\ÜÛœÙKœİ]\×ØÛÙHOHBˆ\ÜÙ\”][İHˆ[ˆ™\ÜÛœÙKšœÛÛŠ
VÈ™]Z[—Bˆ\ÜÙ\™˜Xİ\˜][Ûˆˆ[ˆ™\ÜÛœÙKšœÛÛŠ
VÈ™]Z[—B‚‚™Yˆ\İØ\ÜÚ\İ[Ù^ÜÙ\×Ü™]WØY\—Ù›Ü—İ[\Ü˜\WÜ˜]WÛ[Z]
ÛY[[ÛšÙ^\]Ú
N‚ˆÛ\ÜÈ˜ZÙT™\ÜÛœÙN‚ˆİ]\×ØÛÙHHBˆXY\œÈHÈ”™]KPY\ˆˆŒLˆŸB‚ˆİ]XÛY]ÙˆYˆœÛÛŠ
N‚ˆ™]\›ˆÈ™\œ›ÜˆˆÈ\Hˆœ˜]WÛ[Z]Ù\œ›Üˆ‹˜ÛÙHˆœÛİ×ÙİÛˆŸ_B‚ˆ\Ş[˜ÈYˆ˜ZÙWÜÜİ
Ù[‹\›
ŠšİØ\™ÜÊN‚ˆ™]\›ˆ˜ZÙT™\ÜÛœÙJ
B‚ˆ[ÛšÙ^\]ÚœÙ]]Š\Ş[˜ĞÛY[œÜİ‹˜ZÙWÜÜİ
Bˆ™\ÜÛœÙHHÛY[œÜİ
ˆ‹İŒKØ\ÜÚ\İ[‹ˆXY\œÏ^È]]Üš^˜][Ûˆˆ™X\™\ˆ\İ\ÙXÜ™]]ÚÙ[ˆŸKˆœÛÛ^Èœ›Û\ˆ›Ûš›İ\ˆŸKˆ
Bˆ\ÜÙ\™\ÜÛœÙKœİ]\×ØÛÙHOHBˆ\ÜÙ\œÛİ×ÙİÛˆˆ[ˆ™\ÜÛœÙKšœÛÛŠ
VÈ™]Z[—Bˆ\ÜÙ\™\ÜÛœÙKšXY\œÖÈœ™]KXY\ˆ—HOHŒLˆ‚‚‚™Yˆ\İØ\ÜÚ\İ[Ù[™›Ü˜Ù\×Ü˜]WÛ[Z]
ÛY[[ÛšÙ^\]Ú
N‚ˆ\Ş[˜ÈYˆ˜ZÙWÜÜİ
Ù[‹\›
ŠšİØ\™ÜÊN‚ˆÛ\ÜÈ˜ZÙT™\ÜÛœÙN‚ˆİ]\×ØÛÙHHŒ‚ˆİ]XÛY]ÙˆYˆœÛÛŠ
N‚ˆ™]\›ˆÈ›İ]]ˆŞÈ\Hˆ›Y\ÜØYÙH‹˜ÛÛ[ˆŞÈ\Hˆ›İ]]İ^‹^ˆ“ÒÈŸW_W_B‚ˆ™]\›ˆ˜ZÙT™\ÜÛœÙJ
B‚ˆ[ÛšÙ^\]ÚœÙ]]Š\Ş[˜ĞÛY[œÜİ‹˜ZÙWÜÜİ
Bˆ[ÛšÙ^\]ÚœÙ]]Š\K“PVÔ‘TUQTÕ×ÔT—ÕÒS‘ÕÈ‹JBˆš\œİHÛY[œÜİ
‹İŒKØ\ÜÚ\İ[‹XY\œÏ^È]]Üš^˜][Ûˆˆ™X\™\ˆ\İ\ÙXÜ™]]ÚÙ[ˆŸKœÛÛ^Èœ›Û\ˆŒHŸJBˆÙXÛÛ™HÛY[œÜİ
‹İŒKØ\ÜÚ\İ[‹XY\œÏ^È]]Üš^˜][Ûˆˆ™X\™\ˆ\İ\ÙXÜ™]]ÚÙ[ˆŸKœÛÛ^Èœ›Û\ˆŒˆŸJBˆ\ÜÙ\š\œİœİ]\×ØÛÙHOHŒˆ\ÜÙ\ÙXÛÛ™œİ]\×ØÛÙHOHB‚