from __future__ import annotations

import base64

from app.util import hashed_vector
from tests.conftest import TINY_PNG


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_bad_password(client):
    r = client.post("/api/v1/auth/token", json={"username": "athlete", "password": "nope"})
    assert r.status_code == 401
    assert r.json()["status"] == "error"


def test_auth_required(client):
    r = client.get("/api/v1/hospitals/nearby", params={"latitude": 11.66, "longitude": 78.14})
    assert r.status_code == 401


def test_injury_analyze(client):
    from tests.conftest import _auth

    r = client.post(
        "/api/v1/injury/analyze",
        headers=_auth(client),
        json={
            "image": TINY_PNG,
            "body_part": "right_knee",
            "pain_level": 8,
            "symptoms": ["swelling", "unable_to_walk"],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["injury_type"] == "right_knee_trauma"
    assert len(data["image_features"]) == 512
    assert data["detected_indicators"]["swelling"] > 0.5
    assert "disclaimer" in r.json()


def test_injury_invalid_image(client):
    from tests.conftest import _auth

    r = client.post(
        "/api/v1/injury/analyze",
        headers=_auth(client),
        json={
            "image": base64.b64encode(b"not-an-image").decode(),
            "body_part": "knee",
            "pain_level": 3,
            "symptoms": [],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_symptoms_analyze(client):
    from tests.conftest import _auth

    r = client.post(
        "/api/v1/symptoms/analyze",
        headers=_auth(client),
        json={
            "symptom_text": "Right knee pain, 8/10, unable to walk, swelling present",
            "conversation_history": [
                {"role": "user", "content": "Where is the pain?"},
                {"role": "assistant", "content": "Right knee"},
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data["symptom_embedding"]) == 768
    assert data["extracted_entities"]["location"] == "right_knee"
    assert data["extracted_entities"]["severity"] == 8
    assert data["extracted_entities"]["functional_impairment"] == "unable_to_walk"
    assert "swelling" in data["extracted_entities"]["visible_signs"]
    assert data["suggested_questions"]


def test_vitals_estimate_fallback(client):
    from tests.conftest import _auth

    blob = base64.b64encode(b"ftypmp42" + b"\x00" * 120).decode()
    r = client.post(
        "/api/v1/vitals/estimate",
        headers=_auth(client),
        json={"video": blob, "duration_sec": 30},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["heart_rate"]["unit"] == "BPM"
    assert len(data["ppg_signal"]) == 900
    assert r.json()["disclaimer"]


def test_vitals_too_small(client):
    from tests.conftest import _auth

    r = client.post(
        "/api/v1/vitals/estimate",
        headers=_auth(client),
        json={"video": base64.b64encode(b"tiny").decode(), "duration_sec": 30},
    )
    assert r.status_code == 400


def test_risk_assess(client):
    from tests.conftest import _auth

    body = {
        "image_features": hashed_vector("img", 512),
        "symptom_embedding": hashed_vector("txt", 768),
        "vital_features": hashed_vector("vit", 256),
        "athlete_profile": {
            "age": 22,
            "medical_history": ["asthma"],
            "previous_injuries": ["ankle_sprain"],
        },
    }
    r = client.post("/api/v1/risk/assess", headers=_auth(client), json=body)
    assert r.status_code == 200
    data = r.json()["data"]
    assert 0 <= data["risk_score"] <= 100
    assert data["severity"] in {"LOW", "MEDIUM", "HIGH"}
    assert data["model_version"]
    assert len(data["top_indicators"]) == 3
    # cache hit
    r2 = client.post("/api/v1/risk/assess", headers=_auth(client), json=body)
    assert r2.status_code == 200


def test_risk_wrong_dim(client):
    from tests.conftest import _auth

    r = client.post(
        "/api/v1/risk/assess",
        headers=_auth(client),
        json={
            "image_features": [0.1] * 10,
            "symptom_embedding": hashed_vector("t", 768),
            "vital_features": hashed_vector("v", 256),
        },
    )
    assert r.status_code == 400


def test_hospitals_nearby(client):
    from tests.conftest import _auth

    r = client.get(
        "/api/v1/hospitals/nearby",
        headers=_auth(client),
        params={"latitude": 11.6643, "longitude": 78.1460, "radius_km": 10},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["total_count"] >= 1
    assert data["hospitals"][0]["id"]
    assert "distance_km" in data["hospitals"][0]


def test_incident_lifecycle_and_alert(client):
    from tests.conftest import _auth

    h = _auth(client)
    created = client.post(
        "/api/v1/incident/create",
        headers=h,
        json={
            "athlete_id": "ath_001",
            "injury_photo_url": "https://storage.googleapis.com/example.jpg",
            "body_part": "right_knee",
            "pain_level": 8,
            "symptoms": ["swelling", "severe_pain"],
            "vital_signs": {"heart_rate": 96, "spo2": 98, "respiratory_rate": 18},
            "risk_assessment": {"risk_score": 82, "severity": "HIGH", "confidence": 0.94},
            "location": {"lat": 11.6643, "lng": 78.1460},
            "first_aid_provided": True,
            "hospital_selected": "hosp_001",
        },
    )
    assert created.status_code == 200, created.text
    iid = created.json()["data"]["incident_id"]
    assert created.json()["data"]["timeline"]

    alert = client.post(
        "/api/v1/emergency/alert",
        headers=h,
        json={
            "incident_id": iid,
            "contact_ids": ["coach_001", "parent_001"],
            "message_template": "emergency_high_risk",
            "location": {"lat": 11.6643, "lng": 78.1460},
            "incident_summary": {
                "athlete_name": "Player 07",
                "injury_type": "Right Knee",
                "risk_score": 82,
                "severity": "HIGH",
            },
        },
    )
    assert alert.status_code == 200
    assert alert.json()["data"]["alerts_sent"] == 2

    got = client.get(f"/api/v1/incident/{iid}", headers=h)
    assert got.status_code == 200
    detail = got.json()["data"]
    assert detail["athlete"]["name"] == "Player 07"
    assert detail["injury"]["body_part"] == "right_knee"
    assert any(a.get("action") == "emergency_contact_notified" for a in detail["actions_taken"])


def test_incident_not_found(client):
    from tests.conftest import _auth

    r = client.get("/api/v1/incident/inc_missing", headers=_auth(client, "clinician", "clinician-demo"))
    assert r.status_code == 404


def test_athlete_cannot_create_for_other(client):
    from tests.conftest import _auth

    r = client.post(
        "/api/v1/incident/create",
        headers=_auth(client),
        json={"athlete_id": "ath_someone_else", "symptoms": ["pain"]},
    )
    assert r.status_code == 403


def test_openapi_exists(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/injury/analyze" in paths
    assert "/api/v1/risk/assess" in paths
