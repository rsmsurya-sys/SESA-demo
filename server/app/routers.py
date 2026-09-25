from __future__ import annotations

from fastapi import APIRouter, Query

from app.deps import AuthUser, CacheDep, SettingsDep, StoreDep
from app.exceptions import ForbiddenError
from app.inference import analyze_injury, analyze_symptoms, assess_risk, estimate_vitals
from app.schemas import (
    EmergencyAlertRequest,
    Envelope,
    IncidentCreateRequest,
    InjuryAnalyzeRequest,
    RiskAssessRequest,
    SymptomAnalyzeRequest,
    TokenRequest,
    TokenResponse,
    VitalsEstimateRequest,
)
from app.auth import authenticate_demo, create_access_token
from app.util import haversine_km, iso, travel_min

DISCLAIMER_VITALS = "Estimated values, not medical-grade"
DISCLAIMER_GENERAL = (
    "Not a diagnostic tool. Emergency triage assistance only. Call emergency services "
    "if the person is unconscious, not breathing, or bleeding heavily."
)

router = APIRouter()
auth_router = APIRouter()


@auth_router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def issue_token(body: TokenRequest, settings: SettingsDep) -> TokenResponse:
    sub, role = authenticate_demo(settings, body.username, body.password)
    token = create_access_token(settings, sub=sub, role=role)
    return TokenResponse(
        access_token=token,
        role=role,
        expires_in_minutes=settings.jwt_expire_minutes,
    )


@router.post("/injury/analyze", tags=["injury"])
def injury_analyze(body: InjuryAnalyzeRequest, user: AuthUser, settings: SettingsDep) -> Envelope:
    _ = user
    data = analyze_injury(
        image_b64=body.image,
        body_part=body.body_part,
        pain_level=body.pain_level,
        symptoms=body.symptoms,
        settings=settings,
    )
    return Envelope(data=data, disclaimer=DISCLAIMER_GENERAL)


@router.post("/symptoms/analyze", tags=["symptoms"])
def symptoms_analyze(body: SymptomAnalyzeRequest, user: AuthUser) -> Envelope:
    _ = user
    data = analyze_symptoms(body.symptom_text, body.conversation_history)
    return Envelope(data=data, disclaimer=DISCLAIMER_GENERAL)


@router.post("/vitals/estimate", tags=["vitals"])
def vitals_estimate(body: VitalsEstimateRequest, user: AuthUser, settings: SettingsDep) -> Envelope:
    _ = user
    data = estimate_vitals(body.video, body.duration_sec, settings)
    return Envelope(data=data, disclaimer=DISCLAIMER_VITALS)


@router.post("/risk/assess", tags=["risk"])
def risk_assess(
    body: RiskAssessRequest,
    user: AuthUser,
    settings: SettingsDep,
    cache: CacheDep,
) -> Envelope:
    _ = user
    key = f"risk:{round(sum(body.image_features[:8]), 5)}:{round(sum(body.symptom_embedding[:8]), 5)}"
    cached = cache.get(key)
    if cached is not None:
        return Envelope(data=cached, disclaimer=DISCLAIMER_GENERAL)
    profile = body.athlete_profile.model_dump() if body.athlete_profile else None
    data = assess_risk(
        body.image_features,
        body.symptom_embedding,
        body.vital_features,
        profile,
        settings,
    )
    cache.set(key, data, settings.cache_ttl_seconds)
    return Envelope(data=data, disclaimer=DISCLAIMER_GENERAL)


@router.get("/hospitals/nearby", tags=["hospitals"])
def hospitals_nearby(
    user: AuthUser,
    store: StoreDep,
    cache: CacheDep,
    settings: SettingsDep,
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10, ge=0.5, le=100),
) -> Envelope:
    _ = user
    ckey = f"hosp:{round(latitude, 3)}:{round(longitude, 3)}:{radius_km}"
    cached = cache.get(ckey)
    if cached is not None:
        return Envelope(data=cached)
    rows = []
    for h in store.hospitals:
        d = haversine_km(latitude, longitude, h["lat"], h["lng"])
        if d <= radius_km:
            rows.append(
                {
                    "id": h["id"],
                    "name": h["name"],
                    "distance_km": round(d, 2),
                    "estimated_travel_min": travel_min(d),
                    "emergency_services": h["emergency_services"],
                    "specialties": h["specialties"],
                    "rating": h["rating"],
                    "contact": h["contact"],
                    "location": {"lat": h["lat"], "lng": h["lng"]},
                }
            )
    rows.sort(key=lambda r: (not r["emergency_services"], r["distance_km"]))
    data = {"hospitals": rows, "total_count": len(rows)}
    cache.set(ckey, data, settings.cache_ttl_seconds)
    return Envelope(data=data)


@router.post("/emergency/alert", tags=["emergency"])
def emergency_alert(body: EmergencyAlertRequest, user: AuthUser, store: StoreDep) -> Envelope:
    _ = user
    when = iso()
    statuses = [
        {"contact_id": cid, "status": "delivered", "timestamp": when}
        for cid in body.contact_ids
    ]
    from app.exceptions import NotFoundError

    try:
        store.get_incident(body.incident_id)
    except NotFoundError:
        pass
    store.append_alert(body.incident_id, body.contact_ids, when)
    return Envelope(
        data={
            "alerts_sent": len(statuses),
            "delivery_status": statuses,
            "notification_channels": ["push", "sms"],
        }
    )


@router.post("/incident/create", tags=["incidents"])
def incident_create(body: IncidentCreateRequest, user: AuthUser, store: StoreDep) -> Envelope:
    if user.role == "athlete" and body.athlete_id != user.sub:
        raise ForbiddenError("Athletes can only create incidents for themselves")
    payload = body.model_dump()
    if payload.get("location"):
        loc = payload["location"]
        payload["location"] = {"lat": loc["lat"], "lng": loc["lng"]}
    doc = store.create_incident(payload)
    return Envelope(
        data={
            "incident_id": doc["incident_id"],
            "created_at": doc["created_at"],
            "timeline": doc["timeline"],
        }
    )


@router.get("/incident/{incident_id}", tags=["incidents"])
def incident_get(incident_id: str, user: AuthUser, store: StoreDep) -> Envelope:
    doc = store.get_incident(incident_id)
    if user.role == "athlete" and doc.get("athlete_id") != user.sub:
        raise ForbiddenError("Not allowed to view this incident")
    athlete = store.get_athlete(doc["athlete_id"])
    return Envelope(
        data={
            "incident_id": doc["incident_id"],
            "athlete": {
                "name": athlete.get("name"),
                "age": athlete.get("age"),
                "medical_history": athlete.get("medical_history") or [],
            },
            "injury": {
                "body_part": doc.get("body_part"),
                "photo_url": doc.get("injury_photo_url"),
                "pain_level": doc.get("pain_level"),
                "symptoms": doc.get("symptoms") or [],
            },
            "vital_signs": doc.get("vital_signs"),
            "risk_assessment": doc.get("risk_assessment"),
            "timeline": doc.get("timeline") or [],
            "actions_taken": doc.get("actions_taken") or [],
            "location": doc.get("location"),
        }
    )
