"""Optional bearer-token check for services exposed on the network.

On the Orin every service listens on loopback, so this stays off. The A6000
box is reachable from the LAN, and an open /transcribe there is an open
microphone-transcription endpoint for anyone on the network.

Set KOTONOHA_SERVICE_TOKEN on the service and KOTONOHA__REMOTE__TOKEN on the
orchestrator. With the variable unset the check is disabled and that fact is
logged at startup, so an unprotected service is never a silent surprise.

This is a shared-secret check on a trusted LAN, nothing more. It is not a
substitute for keeping the box off the public internet.
"""

from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..logging_setup import get_logger

log = get_logger(__name__)

ENV_VAR = "KOTONOHA_SERVICE_TOKEN"
# /health stays open so a load balancer or `netcheck` can see the service.
OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


def install_auth(app: FastAPI, service: str) -> None:
    token = os.environ.get(ENV_VAR, "").strip()
    if not token:
        log.warning("auth.disabled", service=service, hint=f"set {ENV_VAR} to require a token")
        return

    log.info("auth.enabled", service=service)

    @app.middleware("http")
    async def _check(request: Request, call_next):  # noqa: ANN001
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        # compare_digest, so a wrong token cannot be found one byte at a time.
        if scheme.lower() != "bearer" or not hmac.compare_digest(presented.strip(), token):
            log.warning("auth.rejected", service=service, path=request.url.path)
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)
