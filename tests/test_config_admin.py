"""Remote configuration management API and client."""

from __future__ import annotations

import httpx
import pytest
import yaml
from fastapi import FastAPI

from kotonoha.clients.config_admin import RemoteConfigClient
from kotonoha.config import load_settings
from kotonoha.services.auth import install_auth
from kotonoha.services.config_admin import router


@pytest.fixture
def admin_environment(tmp_path, monkeypatch):
    override = tmp_path / "remote-server.local.yaml"
    llm_environment = tmp_path / "remote-llm.env"
    monkeypatch.setenv("KOTONOHA_CONFIG", "config/remote-server.yaml")
    monkeypatch.setenv("KOTONOHA_LOCAL_CONFIG", str(override))
    monkeypatch.setenv("KOTONOHA_LLM_CONFIG_ENV", str(llm_environment))
    monkeypatch.setenv("KOTONOHA_SERVICE_TOKEN", "test-secret")
    return override, llm_environment


def build_admin_app() -> FastAPI:
    application = FastAPI()
    install_auth(application, "test-admin")
    application.include_router(router)
    return application


async def test_admin_api_requires_the_service_token(admin_environment):
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/admin/config")).status_code == 401
        response = await client.get(
            "/admin/config", headers={"authorization": "Bearer test-secret"}
        )
        assert response.status_code == 200


async def test_admin_api_validates_and_persists_overrides(admin_environment):
    override, llm_environment = admin_environment
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": "Bearer test-secret"},
    ) as client:
        response = await client.put(
            "/admin/config",
            json={"changes": {"llm.n_ctx": 8192, "asr_verify.compute_type": "float16"}},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["llm"]["n_ctx"] == 8192
    assert "llm.n_ctx" in body["editable_paths"]
    assert "remote.token" not in body["editable_paths"]
    assert body["restart_required"] is True

    written = yaml.safe_load(override.read_text(encoding="utf-8"))
    assert written["llm"]["n_ctx"] == 8192
    assert written["asr_verify"]["compute_type"] == "float16"
    environment = llm_environment.read_text(encoding="utf-8")
    assert "export LLM_PROFILE=moe" in environment
    assert "export LLM_CTX=8192" in environment
    assert "export LLM_BATCH=512" in environment
    assert "export LLM_MODEL=/models/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" in environment


async def test_admin_api_rejects_invalid_values_without_writing(admin_environment):
    override, _ = admin_environment
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": "Bearer test-secret"},
    ) as client:
        response = await client.put(
            "/admin/config", json={"changes": {"llm.n_ctx": "many"}}
        )
    assert response.status_code == 422
    assert not override.exists()


async def test_admin_api_rejects_client_owned_settings(admin_environment):
    override, _ = admin_environment
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": "Bearer test-secret"},
    ) as client:
        response = await client.put(
            "/admin/config", json={"changes": {"remote.token": "replacement"}}
        )
    assert response.status_code == 422
    assert "remote.token" in response.text
    assert not override.exists()


async def test_remote_config_client_reads_and_updates(admin_environment):
    application = build_admin_app()
    settings = load_settings("config/performance.yaml")
    settings.remote.token = "test-secret"
    client = RemoteConfigClient("http://remote.test", settings.remote)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://remote.test",
        headers={"authorization": "Bearer test-secret"},
    )
    try:
        before = await client.read()
        assert before.config["llm"]["profile"] == "moe"
        assert "tts.chunk_ms" in before.editable_paths
        after = await client.update({"tts.chunk_ms": 320})
        assert after.config["tts"]["chunk_ms"] == 320
        assert after.overrides["tts"]["chunk_ms"] == 320
    finally:
        await client.aclose()
