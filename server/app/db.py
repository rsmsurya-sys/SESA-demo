"""Document store: in-memory by default, optional MongoDB.

Firestore-shaped collections:
  incidents, alerts, athletes
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from app.config import Settings
from app.exceptions import NotFoundError
from app.util import iso, new_id


SEED_ATHLETES = {
    "ath_001": {
        "id": "ath_001",
        "name": "Player 07",
        "age": 22,
        "medical_history": ["asthma"],
        "previous_injuries": ["ankle_sprain"],
    }
}

SEED_HOSPITALS = [
    {
        "id": "hosp_valli",
        "name": "Valli Super Speciality Hospital (Valli Orthopedic & Sports Hospital)",
        "lat": 11.6660,
        "lng": 78.1360,
        "emergency_services": True,
        "specialties": ["sports_medicine", "arthroscopy", "orthopedics", "trauma"],
        "rating": 4.9,
        "contact": "+91 90034 17111",
    },
    {
        "id": "hosp_manipal",
        "name": "Manipal Hospital Salem — Emergency & Trauma Centre",
        "lat": 11.6850,
        "lng": 78.1250,
        "emergency_services": True,
        "specialties": ["emergency", "trauma", "icu", "cardiology"],
        "rating": 4.8,
        "contact": "+91 427 234 6666",
    },
    {
        "id": "hosp_gmkmch",
        "name": "Government Mohan Kumaramangalam Medical College & Hospital",
        "lat": 11.6583,
        "lng": 78.1585,
        "emergency_services": True,
        "specialties": ["trauma", "orthopedics", "emergency", "blood_bank"],
        "rating": 4.6,
        "contact": "+91 427 221 1200",
    },
    {
        "id": "hosp_gokulam",
        "name": "Sri Gokulam Hospital & Research Institute",
        "lat": 11.6620,
        "lng": 78.1410,
        "emergency_services": True,
        "specialties": ["orthopedics", "emergency", "spine_surgery"],
        "rating": 4.7,
        "contact": "+91 427 244 8171",
    },
    {
        "id": "hosp_kauvery",
        "name": "Kauvery Hospital Salem",
        "lat": 11.6910,
        "lng": 78.1180,
        "emergency_services": True,
        "specialties": ["neuro_trauma", "cardiac_emergency", "icu"],
        "rating": 4.7,
        "contact": "+91 427 277 7000",
    },
    {
        "id": "hosp_sks",
        "name": "SKS Hospital & Post Graduate Medical Institute",
        "lat": 11.6700,
        "lng": 78.1380,
        "emergency_services": True,
        "specialties": ["orthopedics", "joint_replacement", "trauma"],
        "rating": 4.5,
        "contact": "+91 427 404 1111",
    },
]


class Store:
    def __init__(self) -> None:
        self._lock = Lock()
        self.incidents: dict[str, dict[str, Any]] = {}
        self.alerts: list[dict[str, Any]] = []
        self.athletes = dict(SEED_ATHLETES)
        self.hospitals = list(SEED_HOSPITALS)

    def create_incident(self, payload: dict[str, Any]) -> dict[str, Any]:
        iid = new_id("inc")
        now = iso()
        timeline = [{"event": "injury_reported", "timestamp": now}]
        if payload.get("vital_signs"):
            timeline.append({"event": "vitals_checked", "timestamp": now})
        if payload.get("risk_assessment"):
            timeline.append({"event": "risk_assessed", "timestamp": now})
        if payload.get("first_aid_provided"):
            timeline.append(
                {
                    "event": "first_aid_guidance_displayed",
                    "timestamp": now,
                }
            )
        if payload.get("hospital_selected"):
            timeline.append(
                {
                    "event": "hospital_selected",
                    "timestamp": now,
                    "hospital_id": payload["hospital_selected"],
                }
            )
        doc = {
            "incident_id": iid,
            "created_at": now,
            "athlete_id": payload["athlete_id"],
            "injury_photo_url": payload.get("injury_photo_url"),
            "body_part": payload.get("body_part"),
            "pain_level": payload.get("pain_level"),
            "symptoms": payload.get("symptoms") or [],
            "vital_signs": payload.get("vital_signs"),
            "risk_assessment": payload.get("risk_assessment"),
            "location": payload.get("location"),
            "first_aid_provided": payload.get("first_aid_provided", False),
            "hospital_selected": payload.get("hospital_selected"),
            "timeline": timeline,
            "actions_taken": [
                e
                for e in [
                    {"action": "first_aid_guidance_displayed", "timestamp": now}
                    if payload.get("first_aid_provided")
                    else None,
                    {
                        "action": "hospital_selected",
                        "hospital_id": payload.get("hospital_selected"),
                        "timestamp": now,
                    }
                    if payload.get("hospital_selected")
                    else None,
                ]
                if e
            ],
        }
        with self._lock:
            self.incidents[iid] = doc
        return doc

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        with self._lock:
            doc = self.incidents.get(incident_id)
        if not doc:
            raise NotFoundError(f"Incident not found: {incident_id}")
        return doc

    def append_alert(self, incident_id: str, contact_ids: list[str], when: str) -> None:
        with self._lock:
            self.alerts.append(
                {"incident_id": incident_id, "contact_ids": contact_ids, "timestamp": when}
            )
            inc = self.incidents.get(incident_id)
            if inc is not None:
                inc["timeline"].append({"event": "contacts_notified", "timestamp": when})
                for cid in contact_ids:
                    inc["actions_taken"].append(
                        {
                            "action": "emergency_contact_notified",
                            "contact_id": cid,
                            "timestamp": when,
                        }
                    )

    def get_athlete(self, athlete_id: str) -> dict[str, Any]:
        return self.athletes.get(athlete_id) or {
            "id": athlete_id,
            "name": athlete_id,
            "age": None,
            "medical_history": [],
        }


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def reset_store() -> Store:
    global _store
    _store = Store()
    return _store


def try_mongo(settings: Settings) -> Store | None:
    """Optional Mongo persistence; returns None to keep in-memory store."""
    if not settings.mongo_uri:
        return None
    return None  # Wire Motor/pymongo here when a cluster is provisioned.
