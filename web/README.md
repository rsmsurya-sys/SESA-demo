# SESA MedAI™ — Clinical Emergency Sports Trauma & Triage System

**Autonomous Multi-Modal Clinical Decision Support System (CDSS) for Sports Field Emergencies & Trauma Care.**  
*Complies with IEC 62304 / ISO 14971 Medical Software Architectures.*

---

## System Capabilities

- **Computer Vision Lesion Segmentation**: EfficientNet-B4 dense visual feature extraction with Grad-CAM explainability heatmaps for joint effusion, hemarthrosis, and structural deformity.
- **Clinical NLP Phenotyping**: Bio-ClinicalBERT token encoding with voice dictation support (English & Tamil) for anamnesis and Ottawa rules evaluation.
- **Contactless rPPG Biosensor Telemetry**: Non-invasive green-channel chrominance spectrophotometry estimating Heart Rate (HR), SpO₂, Respiratory Rate (RR), Perfusion Index, and Shock Index (SI).
- **Cross-Modal Transformer Fusion**: 4-Layer Self-Attention Transformer mapping multimodal tokens to Emergency Severity Index (ESI Levels 1–5) and Manchester Triage tiers.
- **Physician Decision Support**: Auto-generated clinical SOAP notes with 1-click clipboard export and printable official hospital incident report sheets.
- **Emergency Dispatch & Geolocation**: Direct 108 / 112 emergency hotline speed-dial, live GPS coordinate encoding, and 1-click WhatsApp SOS dispatch.
- **Bilingual CPR Rhythm Metronome**: Interactive 110 BPM American Heart Association cardiac rhythm metronome with voice-guided instructions in Tamil and English.

---

## Quick Launch

### Option 1 — Direct Browser Launch
Open `web/index.html` directly in any modern web browser (Chrome, Edge, Safari, Firefox).

### Option 2 — Local HTTP Server (Recommended)
```powershell
# From workspace root
python -m http.server 8080 --directory web
```
Navigate to: `http://localhost:8080`

### Option 3 — FastAPI Backend Integration
```powershell
# Terminal 1: Launch FastAPI CDSS Service
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir server --reload --port 5674

# Terminal 2: Serve Web Console
python -m http.server 8080 --directory web
```

---

## Clinical Roles & Access

| Duty Role | Security Credential | Permissions |
|---|---|---|
| **Emergency Physician / Team Doctor** | `er-physician-auth` | Full diagnostic telemetry, SOAP note export, ESI triage review |
| **Licensed Sports Paramedic** | `paramedic-auth` | Field trauma triage, rPPG acquisition, 108 trauma bay dispatch |
| **Certified Athletic Trainer / Coach** | `coach-first-aid` | P.O.L.I.C.E protocol execution, CPR guidance, parent/team alert |
| **Athlete Self-Reporting** | `athlete-auth` | Symptom logging, pain VAS rating, nearest emergency routing |

---

## System Architecture

```
[ Field Camera / Image ]  ──> EfficientNet-B4  ──> 512-d Vision Vector   ┐
[ Speech / Clinical Note] ──> Bio-ClinicalBERT ──> 768-d NLP Embedding   ┼─> 4-Layer Cross-Attention ──> ESI Acuity / SOAP Note / 108 Dispatch
[ Optical rPPG Video ]    ──> VitalCNN-1D      ──> 256-d Biosensor Token ┘
```
