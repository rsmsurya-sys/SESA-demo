from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    stamp = utcnow().strftime("%Y%m%d")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def hashed_vector(seed: str, dim: int) -> list[float]:
    """Deterministic unit vector from a string (no PHI in the seed if you can avoid it)."""
    out: list[float] = []
    block = seed.encode("utf-8")
    while len(out) < dim:
        block = hashlib.sha256(block).digest()
        for i in range(0, 32, 4):
            unsigned = int.from_bytes(block[i : i + 4], "big")
            out.append((unsigned / 2**32) * 2.0 - 1.0)
            if len(out) >= dim:
                break
    return l2_normalize(out[:dim])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_min(distance_km: float, kmh: float = 28.0) -> int:
    return max(1, int(round(distance_km / kmh * 60.0)))


def redact(obj: Any) -> Any:
    """Drop bulky/sensitive fields before logging."""
    if isinstance(obj, dict):
        blocked = {
            "image",
            "video",
            "image_features",
            "symptom_embedding",
            "vital_features",
            "ppg_signal",
            "password",
        }
        return {k: ("<omitted>" if k in blocked else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list) and len(obj) > 32:
        return f"<list n={len(obj)}>"
    return obj
