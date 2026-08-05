"""Remote configuration management API and client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from kotonoha._config import load_settings
from kotonoha.clients._config_admin import RemoteConfigClient
from kotonoha.services._auth import install_auth
from kotonoha.services._config_admin import router


@pytest.fixture
def admin_environment(
    _positional_only: object | None = None,
    /,
    *,
    tmp_path: Any,
    monkeypatch: Any,
) -> Any:
    override = tmp_path / "remote-server.local.yaml"
    monkeypatch.setenv("KOTONOHA_CONFIG", "config/remote-server.yaml")
    monkeypatch.setenv("KOTONOHA_LOCAL_CONFIG", str(override))
    monkeypatch.setenv("KOTONOHA_SERVICE_TOKEN", "test-secret")
    return override


def build_admin_app() -> FastAPI:
    application = FastAPI()
    install_auth(application, "test-admin")
    application.include_router(router)
    return application


async def test_admin_api_requires_the_service_token(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/admin/config")).status_code == 401
        response = await client.get(
            "/admin/config", headers={"authorization": "Bearer test-secret"}
        )
        assert response.status_code == 200


async def test_admin_api_validates_and_persists_overrides(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    override = admin_environment
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": "Bearer test-secret"},
    ) as client:
        response = await client.put(
            "/admin/config",
            json={
                "changes": {
                    "llm.max_model_len": 8192,
                    "asr_verify.compute_type": "float16",
                }
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["llm"]["max_model_len"] == 8192
    assert "llm.max_model_len" in body["editable_paths"]
    assert "asr.vllm_realtime_architecture" in body["editable_paths"]
    assert "remote.token" not in body["editable_paths"]
    assert body["restart_required"] is True

    written = yaml.safe_load(override.read_text(encoding="utf-8"))
    assert written["llm"]["max_model_len"] == 8192
    assert written["asr_verify"]["compute_type"] == "float16"


async def test_admin_api_rejects_invalid_values_without_writing(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    override = admin_environment
    transport = httpx.ASGITransport(app=build_admin_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": "Bearer test-secret"},
    ) as client:
        response = await client.put(
            "/admin/config", json={"changes": {"llm.max_model_len": "many"}}
        )
    assert response.status_code == 422
    assert not override.exists()


async def test_admin_api_rejects_client_owned_settings(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    override = admin_environment
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


async def test_remote_config_client_reads_and_updates(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
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
        assert before.config["llm"]["profile"] == "translategemma"
        assert before.config["llm"]["profiles"]["translategemma"]["directory"] == (
            "translategemma-12b-it"
        )
        assert "llm.max_num_seqs" in before.editable_paths
        assert not any(path.startswith("tts.") for path in before.editable_paths)
        after = await client.update({"llm.max_num_seqs": 2})
        assert after.config["llm"]["max_num_seqs"] == 2
        assert after.overrides["llm"]["max_num_seqs"] == 2
    finally:
        await client.aclose()
