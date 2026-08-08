"""SQLite configuration management API contracts."""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest
from fastapi import FastAPI

from kotonoha._config import configuration_database_path, load_settings
from kotonoha._config_store import apply_changes
from kotonoha.services._auth import install_auth
from kotonoha.services._config_admin import router

SERVICE_TOKEN = "test-service-token-0123456789abcdef"


def build_admin_app() -> FastAPI:
    application = FastAPI()
    install_auth(application, "test-admin")
    application.include_router(router)

    @application.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    return application


@pytest.fixture
def authenticated_environment(
    _positional_only: object | None = None,
    /,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOTONOHA_SERVICE_TOKEN", SERVICE_TOKEN)


async def test_admin_api_requires_the_service_token(
    _positional_only: object | None = None,
    /,
    *,
    authenticated_environment: None,
) -> None:
    del authenticated_environment
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/admin/config")).status_code == 401
        response = await client.get(
            "/admin/config",
            headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
        )
        assert response.status_code == 200


async def test_admin_api_persists_and_resets_sqlite_overrides(
    _positional_only: object | None = None,
    /,
    *,
    authenticated_environment: None,
) -> None:
    del authenticated_environment
    transport = httpx2.ASGITransport(app=build_admin_app())
    headers = {"authorization": f"Bearer {SERVICE_TOKEN}"}
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers=headers,
    ) as client:
        response = await client.put(
            "/admin/config",
            json={"changes": {"llm.max_model_len": 1536}},
        )
        reset = await client.request(
            "DELETE",
            "/admin/config",
            json={"paths": ["llm.max_model_len"]},
        )

    assert response.status_code == 200
    assert response.json()["config"]["llm"]["max_model_len"] == 1536
    assert Path(response.json()["path"]) == configuration_database_path()
    assert reset.status_code == 200
    assert reset.json()["config"]["llm"]["max_model_len"] == 2048
    assert load_settings().llm.max_model_len == 2048


async def test_admin_api_rejects_invalid_and_web_owned_settings(
    _positional_only: object | None = None,
    /,
    *,
    authenticated_environment: None,
) -> None:
    del authenticated_environment
    transport = httpx2.ASGITransport(app=build_admin_app())
    headers = {"authorization": f"Bearer {SERVICE_TOKEN}"}
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers=headers,
    ) as client:
        invalid = await client.put(
            "/admin/config",
            json={"changes": {"llm.max_model_len": "many"}},
        )
        rejected = await client.put(
            "/admin/config",
            json={"changes": {"transport.token": "replacement"}},
        )

    assert invalid.status_code == 422
    assert rejected.status_code == 422
    assert "transport.token" in rejected.text


async def test_admin_api_redacts_transport_credentials(
    _positional_only: object | None = None,
    /,
    *,
    authenticated_environment: None,
) -> None:
    del authenticated_environment
    assert apply_changes(
        {"transport.token": "persisted-secret-0123456789abcdef"}
    ).written
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        response = await client.get("/admin/config")

    assert response.status_code == 200
    assert "token" not in response.json()["config"]["transport"]
    assert "token" not in response.json()["defaults"]["transport"]
    assert "token" not in response.json()["overrides"].get("transport", {})


async def test_admin_global_reset_preserves_web_owned_overrides(
    _positional_only: object | None = None,
    /,
    *,
    authenticated_environment: None,
) -> None:
    del authenticated_environment
    assert apply_changes(
        {
            "llm.max_model_len": 1536,
            "session.fixed_target": "ja",
        }
    ).written
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        response = await client.request("DELETE", "/admin/config", json={})

    assert response.status_code == 200
    settings = load_settings()
    assert settings.llm.max_model_len == 2048
    assert settings.session.fixed_target == "ja"
