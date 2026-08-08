"""Bound inbound HTTP bodies before multipart parsing allocates storage."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Final

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAXIMUM_REQUEST_BODY_BYTES: Final = 4 * 1024 * 1024


def parse_json_object(
    value: str,
    /,
) -> dict[str, Any]:
    """Parse multipart metadata and reject scalar or array JSON values."""
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


class RequestBodyTooLarge(RuntimeError):
    __slots__: ClassVar[tuple[str, ...]] = ()


class RequestBodyLimitMiddleware:
    """Reject declared and streamed request bodies above a fixed byte limit."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "application",
        "maximum_bytes",
    )
    application: ASGIApp
    maximum_bytes: int

    def __init__(
        self,
        application: ASGIApp,
        /,
        maximum_bytes: int = MAXIMUM_REQUEST_BODY_BYTES,
    ) -> None:
        self.application = application
        self.maximum_bytes = maximum_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        /,
    ) -> None:
        if scope["type"] != "http":
            await self.application(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.maximum_bytes:
            await self._reject(scope, receive, send)
            return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.maximum_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(
            message: Message,
            /,
        ) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.application(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, receive, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        /,
    ) -> None:
        response = PlainTextResponse("request body too large", status_code=413)
        await response(scope, receive, send)


def _content_length(
    scope: Scope,
    /,
) -> int | None:
    for name, value in scope.get("headers", ()):
        if name.lower() != b"content-length":
            continue
        try:
            length = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, length)
    return None
