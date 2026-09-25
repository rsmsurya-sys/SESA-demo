from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["LOW", "MEDIUM", "HIGH"]
Role = Literal["user", "assistant"]


class Envelope(BaseModel):
    status: Literal["success"] = "success"
    data: Any
    disclaimer: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    status: Literal["error"] = "error"
    error: ErrorBody


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class ChatTurn(BaseModel):
    role: Role
    content: str = Field(min_length=1, max_length=2000)


class InjuryAnalyzeRequest(BaseModel):
    image: str = Field(min_length=16, description="Base64 JPEG/PNG (no data: URL required)")
    body_part: str = Field(min_length=2, max_length=64)
    pain_level: int = Field(ge=0, le=10)
    symptoms: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("symptoms")
    @classmethod
    def _norm_symptoms(cls, v: list[str]) -> list[str]:
        return [s.strip().lower().replace(" ", "_") for s in v if s.strip()]


class InjuryAnalyzeData(BaseModel):
    injury_type: str
    detected_indicators: dict[str, float]
    image_features: list[float]
    processing_time_ms: int


class SymptomAnalyzeRequest(BaseModel):
    symptom_text: str = Field(min_length=3, max_length=4000)
    conversation_history: list[ChatTurn] = Field(default_factory=list, max_length=40)


class ExtractedEntities(BaseModel):
    location: str | None = None
    severity: int | None = None
    functional_impairment: str | None = None
    visible_signs: list[str] = Field(default_factory=list)


class SymptomAnalyzeData(BaseModel):
    symptom_embedding: list[float]
    extracted_entities: ExtractedEntities
    suggested_questions: list[str]


class VitalsEstimateRequest(BaseModel):
    video: str = Field(min_length=16)
    duration_sec: int = Field(ge=5, le=60, default=30)


class VitalReading(BaseModel):
    value: float
    unit: str
    confidence: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)


class VitalsEstimateData(BaseModel):
    heart_rate: VitalReading
    spo2: VitalReading
    respiratory_rate: VitalReading
    ppg_signal: list[float]
    processing_time_ms: int


class AthleteProfile(BaseModel):
    age: int = Field(ge=5, le=100)
    medical_history: list[str] = Field(default_factory=list, max_length=30)
    previous_injuries: list[str] = Field(default_factory=list, max_length=30)


class RiskAssessRequest(BaseModel):
    image_features: list[float] = Field(min_length=512, max_length=512)
    symptom_embedding: list[float] = Field(min_length=768, max_length=768)
    vital_features: list[float] = Field(min_length=256, max_length=256)
    athlete_profile: AthleteProfile | None = None


class RiskIndicator(BaseModel):
    factor: str
    weight: float


class RiskAssessData(BaseModel):
    risk_score: float
    severity: Severity
    confidence: float
    top_indicators: list[RiskIndicator]
    recommended_actions: list[str]
    model_version: str


class GeoPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class Hospital(BaseModel):
    id: str
    name: str
    distance_km: float
    estimated_travel_min: int
    emergency_services: bool
    specialties: list[str]
    rating: float
    contact: str
    location: GeoPoint


class HospitalsNearbyData(BaseModel):
    hospitals: list[Hospital]
    total_count: int


class IncidentSummary(BaseModel):
    athlete_name: str
    injury_type: str
    risk_score: float
    severity: Severity


class EmergencyAlertRequest(BaseModel):
    incident_id: str
    contact_ids: list[str] = Field(min_length=1, max_length=10)
    message_template: str = "emergency_high_risk"
    location: GeoPoint
    incident_summary: IncidentSummary


class DeliveryStatus(BaseModel):
    contact_id: str
    status: str
    timestamp: str


class EmergencyAlertData(BaseModel):
    alerts_sent: int
    delivery_status: list[DeliveryStatus]
    notification_channels: list[str]


class VitalSignsDoc(BaseModel):
    heart_rate: float | None = None
    spo2: float | None = None
    respiratory_rate: float | None = None


class RiskDoc(BaseModel):
    risk_score: float
    severity: Severity
    confidence: float


class IncidentCreateRequest(BaseModel):
    athlete_id: str
    injury_photo_url: str | None = None
    body_part: str | None = None
    pain_level: int | None = Field(default=None, ge=0, le=10)
    symptoms: list[str] = Field(default_factory=list)
    vital_signs: VitalSignsDoc | None = None
    risk_assessment: RiskDoc | None = None
    location: GeoPoint | None = None
    first_aid_provided: bool = False
    hospital_selected: str | None = None


class TimelineEvent(BaseModel):
    event: str
    timestamp: str
    hospital_id: str | None = None
    contact_id: str | None = None


class IncidentCreateData(BaseModel):
    incident_id: str
    created_at: str
    timeline: list[TimelineEvent]


class AthletePublic(BaseModel):
    name: str
    age: int | None = None
    medical_history: list[str] = Field(default_factory=list)


class InjuryPublic(BaseModel):
    body_part: str | None = None
    photo_url: str | None = None
    pain_level: int | None = None
    symptoms: list[str] = Field(default_factory=list)


class ActionTaken(BaseModel):
    action: str
    timestamp: str
    hospital_id: str | None = None
    contact_id: str | None = None


class IncidentDetailData(BaseModel):
    incident_id: str
    athlete: AthletePublic
    injury: InjuryPublic
    vital_signs: VitalSignsDoc | None = None
    risk_assessment: RiskDoc | None = None
    timeline: list[TimelineEvent]
    actions_taken: list[ActionTaken]
    location: GeoPoint | None = None
