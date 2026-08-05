from __future__ import annotations

from fastapi import APIRouter

from schemas.dto import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    """Liveness: process is up. Never checks downstream deps — a slow vector
    store shouldn't get this pod killed by the kubelet."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
async def readyz() -> HealthResponse:
    """Readiness: same check for now (no external dep is required to serve
    /chat — the fake providers make everything self-contained). Split out as
    its own endpoint so a future real dependency (e.g. a remote vector DB) has
    somewhere to report into without changing the liveness contract."""
    return HealthResponse(status="ok")
