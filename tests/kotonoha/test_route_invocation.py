"""Framework-invoked callbacks must accept keyword arguments.

FastAPI and Typer build their dependency model from the declared signature and
then invoke the callback with keyword arguments only. A handler declared
positional-only therefore raises at request time, not at import or start-up, so
health checks pass and the failure appears on the first real connection:

    TypeError: realtime_translation() got some positional-only arguments
    passed as keyword arguments: 'websocket'

`keyword_compatible` keeps the positional-only declaration the source standard
asks for and moves the values back into position before dispatch.
"""

from __future__ import annotations

import ast
import pathlib
from typing import ClassVar

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from pydantic import BaseModel

from kotonoha._call_compatibility import keyword_compatible

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "kotonoha"
ROUTE_METHODS = (".get(", ".post(", ".put(", ".delete(", ".patch(", ".websocket(")


def _route_handlers() -> list[tuple[str, str, bool, tuple[str, ...]]]:
    handlers: list[tuple[str, str, bool, tuple[str, ...]]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [ast.unparse(item) for item in node.decorator_list]
            if not any(method in item for item in decorators for method in ROUTE_METHODS):
                continue
            handlers.append(
                (
                    path.relative_to(SOURCE_ROOT).as_posix(),
                    node.name,
                    "keyword_compatible" in decorators,
                    tuple(argument.arg for argument in node.args.posonlyargs),
                )
            )
    return handlers


def test_the_scan_finds_the_service_routes() -> None:
    handlers = _route_handlers()

    assert len(handlers) >= 10
    assert any(name == "realtime_translation" for _path, name, _wrapped, _ in handlers)
    assert any(name == "realtime_transcription" for _path, name, _wrapped, _ in handlers)


def test_every_positional_only_route_handler_is_keyword_compatible() -> None:
    """WebSocket routes were missed once; the deployed service rejected every connection."""
    unwrapped = [
        f"{path}:{name}"
        for path, name, wrapped, positional in _route_handlers()
        if positional and not wrapped
    ]

    assert not unwrapped, (
        "route handlers declare positional-only parameters without "
        f"@keyword_compatible: {unwrapped}"
    )


def test_every_resident_model_service_exposes_live_reload() -> None:
    from kotonoha.services import (
        _asr_server,
        _asr_verify_server,
        _llm_server,
        _tts_server,
    )

    for application in (
        _asr_server.app,
        _asr_verify_server.app,
        _llm_server.app,
        _tts_server.app,
    ):
        assert "/admin/reload" in {
            getattr(route, "path", None) for route in application.routes
        }


class RequestBody(BaseModel):
    __slots__: ClassVar[tuple[str, ...]] = ()
    value: int


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.post("/declared")
    @keyword_compatible
    async def declared(
        body: RequestBody,
        /,
    ) -> dict[str, int]:
        return {"value": body.value}

    @app.websocket("/realtime")
    @keyword_compatible
    async def realtime(
        websocket: WebSocket,
        /,
    ) -> None:
        await websocket.accept()
        await websocket.send_text("accepted")
        await websocket.close()

    return TestClient(app, raise_server_exceptions=False)


def test_a_wrapped_http_handler_accepts_a_request(
    _positional_only: object | None = None,
    /,
    *,
    client: TestClient,
) -> None:
    response = client.post("/declared", json={"value": 7})

    assert response.status_code == 200
    assert response.json() == {"value": 7}


def test_a_wrapped_websocket_handler_accepts_a_connection(
    _positional_only: object | None = None,
    /,
    *,
    client: TestClient,
) -> None:
    with client.websocket_connect("/realtime") as connection:
        assert connection.receive_text() == "accepted"


def test_an_undecorated_handler_still_fails() -> None:
    """Confirms the guard above is testing a real failure mode, not a tautology."""
    app = FastAPI()

    @app.post("/bare")
    async def bare(
        body: RequestBody,
        /,
    ) -> dict[str, int]:
        return {"value": body.value}

    client = TestClient(app, raise_server_exceptions=False)

    assert client.post("/bare", json={"value": 7}).status_code == 500
