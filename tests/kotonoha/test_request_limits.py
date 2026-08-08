"""Inbound service request size limits."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kotonoha._call_compatibility import keyword_compatible
from kotonoha.services._request_limits import (
    RequestBodyLimitMiddleware,
    parse_json_object,
)


def test_declared_oversized_body_is_rejected_before_the_route() -> None:
    application = FastAPI()
    application.add_middleware(RequestBodyLimitMiddleware, maximum_bytes=8)
    called = False

    @application.post("/upload")
    @keyword_compatible
    async def upload(
        request: Request,
        /,
    ) -> dict[str, int]:
        nonlocal called
        called = True
        body = await request.body()
        return {"bytes": len(body)}

    with TestClient(application) as client:
        response = client.post("/upload", content=b"012345678")

    assert response.status_code == 413
    assert not called


def test_body_at_the_limit_reaches_the_route() -> None:
    application = FastAPI()
    application.add_middleware(RequestBodyLimitMiddleware, maximum_bytes=8)

    @application.post("/upload")
    @keyword_compatible
    async def upload(
        request: Request,
        /,
    ) -> dict[str, int]:
        body = await request.body()
        return {"bytes": len(body)}

    with TestClient(application) as client:
        response = client.post("/upload", content=b"01234567")

    assert response.status_code == 200
    assert response.json() == {"bytes": 8}


@pytest.mark.parametrize("value", ("[]", "null", '"text"', "1"))
def test_multipart_parameters_require_a_json_object(
    _positional_only: object | None = None,
    /,
    *,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object(value)
