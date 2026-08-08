"""Remote configuration management API and client."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest
import yaml
from fastapi import FastAPI

from kotonoha._config import load_settings
from kotonoha.clients._base import ServiceApplicationError
from kotonoha.clients._config_admin import RemoteConfigClient
from kotonoha.services._auth import install_auth
from kotonoha.services._config_admin import router

SERVICE_TOKEN = "test-service-token-0123456789abcdef"
REMOTE_CLIENT_TOKEN = "test-remote-token-0123456789abcdef"


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
    monkeypatch.setenv("KOTONOHA_SERVICE_TOKEN", SERVICE_TOKEN)
    monkeypatch.setenv("KOTONOHA__REMOTE__TOKEN", REMOTE_CLIENT_TOKEN)
    return override


def build_admin_app() -> FastAPI:
    application = FastAPI()
    install_auth(application, "test-admin")
    application.include_router(router)

    @application.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    return application


async def test_admin_api_requires_the_service_token(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/admin/config")).status_code == 401
        response = await client.get(
            "/admin/config", headers={"authorization": f"Bearer {SERVICE_TOKEN}"}
        )
        assert response.status_code == 200


async def test_auth_exposes_only_health_without_a_token(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    del admin_environment
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/openapi.json")).status_code == 401
        assert (await client.get("/docs")).status_code == 401


async def test_admin_api_validates_and_persists_overrides(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    override = admin_environment
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
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
    assert "accelerator.profile" in body["editable_paths"]
    assert "logging.prometheus_port" in body["editable_paths"]
    assert "remote.token" not in body["editable_paths"]
    assert "token" not in body["config"]["remote"]
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
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
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
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        response = await client.put(
            "/admin/config", json={"changes": {"remote.token": "replacement"}}
        )
    assert response.status_code == 422
    assert "remote.token" in response.text
    assert not override.exists()


async def test_admin_api_redacts_secrets_from_manual_overrides(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    override = admin_environment
    override.write_text(
        yaml.safe_dump(
            {
                "root": "/sensitive/root",
                "remote": {"token": "manual-secret"},
                "llm": {"max_model_len": 4096},
            }
        ),
        encoding="utf-8",
    )
    transport = httpx2.ASGITransport(app=build_admin_app())
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
    ) as client:
        response = await client.get("/admin/config")

    assert response.status_code == 200
    overrides = response.json()["overrides"]
    assert "root" not in overrides
    assert "token" not in overrides["remote"]
    assert overrides["llm"]["max_model_len"] == 4096


async def test_remote_config_client_reads_and_updates(
    _positional_only: object | None = None,
    /,
    *,
    admin_environment: Any,
) -> None:
    application = build_admin_app()
    settings = load_settings("config/performance.yaml")
    settings.remote.token = SERVICE_TOKEN
    client = RemoteConfigClient("http://remote.test", settings.remote)
    await client._client.aclose()
    client._client = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=application),
        base_url="http://remote.test",
        headers={"authorization": f"Bearer {SERVICE_TOKEN}"},
    )
    try:
        before = await client.read()
        assert before.config["llm"]["profile"] == "translategemma"
        assert before.config["llm"]["profiles"]["translategemma"]["directory"] == (
            "translategemma-12b-it"
        )
        assert "llm.max_num_batched_tokens" in before.editable_paths
        assert "llm.compilation_mode" in before.editable_paths
        assert not any(path.startswith("tts.") for path in before.editable_paths)
        after = await client.update({"llm.max_num_batched_tokens": 8192})
        assert after.config["llm"]["max_num_batched_tokens"] == 8192
        assert after.overrides["llm"]["max_num_batched_tokens"] == 8192
    finally:
        await client.aclose()


async def test_remote_config_client_classifies_streaming_error_without_reading_closed_body(
    _positional_only: object | None = None,
    /,
) -> None:
    settings = load_settings("config/performance.yaml")

    def reject_request(
        request: httpx2.Request,
        /,
    ) -> httpx2.Response:
        del request
        return httpx2.Response(422, content=b'{"detail":"invalid configuration"}')

    client = RemoteConfigClient("http://remote.test", settings.remote)
    await client._client.aclose()
    client._client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(reject_request),
        base_url="http://remote.test",
    )
    try:
        with pytest.raises(ServiceApplicationError, match="HTTP 422"):
            await client.read()
    finally:
        await client.aclose()
