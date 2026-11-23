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


def test_openapi_documents_bearer_authentication(client):
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["BearerAuth"] == {
        "type": "http", "scheme": "bearer"
    }


def test_render_blueprint_builds_from_repository_root():
    with open("render.yaml", encoding="utf-8") as blueprint_file:
        blueprint = yaml.safe_load(blueprint_file)
    service = blueprint["services"][0]
    assert service["dockerfilePath"] == "./Dockerfile"
    assert service["dockerContext"] == "."
    assert {entry["key"] for entry in service["envVars"]} >= {
        "OPENAI_API_KEY", "APP_ACCESS_TOKEN", "OPENAI_MODEL"
    }


def test_assistant_requires_bearer_token(client):
    response = client.post("/v1/assistant", json={"prompt": "Bonjour"})
    assert response.status_code == 401


def test_assistant_rejects_missing_server_configuration(client, monkeypatch):
    monkeypatch.setattr(api, "APP_ACCESS_TOKEN", "")
    response = client.post(
        "/v1/assistant",
        headers={"Authorization": "Bearer test-secret-token"},
        json={"prompt": "Bonjour"},
    )
    assert response.status_code == 503


def test_assistant_rejects_invalid_image_before_provider_call(client, monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("provider must not be called for invalid image")

    monkeypatch.setattr(httpx.AsyncClient, "post", fail_if_called)
    fake_image = base64.b64encode(b"not an image").decode("ascii")
    response = client.post(
        "/v1/assistant",
        headers={"Authorization": "Bearer test-secret-token"},
        json={"prompt": "Décris", "image_data_url": f"data:image/png;base64,{fake_image}"},
    )
    assert response.status_code == 400


def test_assistant_sends_responses_request_and_returns_text(client, monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Bonjour !"}]}]}

    async def fake_post(self, url, **kwargs):
        calls["url"] = url
        calls["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    response = client.post(
        "/v1/assistant",
        headers={"Authorization": "Bearer test-secret-token"},
        json={"prompt": "Bonjour"},
    )
    assert response.status_code == 200
    assert response.json() == {"answer": "Bonjour !", "model": "gpt-6.1-terra"}
    assert calls["url"] == "https://api.openai.com/v1/responses"
    assert calls["json"]["model"] == "gpt-6.1-terra"
    assert calls["json"]["store"] is False


def test_assistant_maps_provider_failure_to_gateway_error(client, monkeypatch):
    class FakeResponse:
        status_code = 401

    async def fake_post(self, url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    response = client.post(
        "/v1/assistant",
        headers={"Authorization": "Bearer test-secret-token"},
        json={"prompt": "Bonjour"},
    )
    assert response.status_code == 502


def test_assistant_enforces_rate_limit(client, monkeypatch):
    async def fake_post(self, url, **kwargs):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(api, "MAX_REQUESTS_PER_WINDOW", 1)
    first = client.post("/v1/assistant", headers={"Authorization": "Bearer test-secret-token"}, json={"prompt": "1"})
    second = client.post("/v1/assistant", headers={"Authorization": "Bearer test-secret-token"}, json={"prompt": "2"})
    assert first.status_code == 200
    assert second.status_code == 429

