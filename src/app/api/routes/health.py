from __future__ import annotations

from fastapi import APIRouter

from schemas.dto import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probu")
async def healthz() -> HealthResponse:
    """Liveness: process ayakta mı. Downstream bağımlılıkları hiç kontrol
    etmiyor — yavaş bir vektör deposu yüzünden bu pod kubelet tarafından
    öldürülmemeli."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probu")
async def readyz() -> HealthResponse:
    """Readiness: şimdilik aynı kontrol (/chat'i servis etmek için dış bir
    bağımlılık gerekmiyor — fake sağlayıcılar her şeyi kendi kendine yeterli
    yapıyor). İleride gerçek bir bağımlılık (ör. uzak bir vektör veritabanı)
    liveness sözleşmesini değiştirmeden buraya rapor edebilsin diye ayrı bir
    endpoint olarak tutuluyor."""
    return HealthResponse(status="ok")
