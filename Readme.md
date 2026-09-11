# AquaGuard AI
## AI-Powered Water & Climate Risk Intelligence Platform

AquaGuard AI is a Streamlit environmental screening and decision-support prototype combining historical climate/weather data, optional water-quality observations, deterministic Python analytics and Generative AI.

**MVP limitation:** the 0–100 scores are heuristic screening indicators, not regulatory classifications, validated probabilities, or substitutes for field/laboratory assessment.

### Workflow
`User inputs → Open-Meteo → validation → climate indicators → optional water-quality indicators → risk engine → AI context → Groq → report → dashboard/downloads`

### Files
- `app.py` — Streamlit application
- `workflow.py` — reusable calculations and end-to-end workflow
- `prompt.py` — centralized AI prompts
- `requirements.txt` — dependencies
- `.env.example` — secret template
- `.gitignore` — protects secrets/local files
- `.streamlit/config.toml` — optional UI theme
- `README.md` — documentation

### Climate calculations
- Temperature stress: `clamp((mean_temperature - 20) / 18 × 100)`
- Dry-day pressure: share of days with precipitation `< 1 mm/day` × 100
- ET0 stress: `clamp(mean_ET0 / 7 × 100)`
- Climate risk: `40% temperature + 35% dry-day + 25% ET0`

### Water quality
Optional columns: `pH`, `turbidity`, `dissolved_oxygen`, `nitrate`, `phosphate`.
Prototype pressure formulas use fixed heuristic reference values. They must be locally calibrated and validated before operational use.

### Overall risk
Without WQ: `Overall = Climate Risk`.
With WQ: `Overall = 55% Climate + 45% Water Quality`.
Bands: `0–33 LOW`, `34–66 MEDIUM`, `67–100 HIGH`.

### AI architecture
The LLM does not calculate the environmental metrics. Python/Pandas/NumPy calculate indicators first; a compact structured context is then sent to Groq for interpretation, recommendations, SDG relevance and data-gap reporting.

### Decision-support value
The goal is `DATA → RISK → EXPLANATION → PRIORITY → ACTION → MONITORING`. Potential users include environmental authorities, local government, water-resource managers, NGOs, researchers and consultants. The system supports decisions; it should not autonomously issue regulatory decisions.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Secrets
Local `.env`:
```text
GROQ_API_KEY=your_real_key
GROQ_MODEL=openai/gpt-oss-20b
```
Streamlit Secrets:
```toml
GROQ_API_KEY = "your_real_key"
GROQ_MODEL = "openai/gpt-oss-20b"
```
Never commit real API keys.

### Deployment
Push `app.py`, `workflow.py`, `prompt.py`, `requirements.txt`, `README.md`, `.env.example`, `.gitignore` and optional `.streamlit/config.toml` to GitHub. Deploy `app.py` on Streamlit Community Cloud and add secrets in the Secrets settings.

### Future roadmap
1. Pakistan-specific PMD/PCRWR/provincial/university datasets where accessible.
2. GIS layers and interactive maps.
3. Sentinel-2/Landsat/MODIS remote sensing (NDVI, NDWI and appropriate water-quality/eutrophication indicators).
4. Historical baselines and anomaly detection.
5. Validated ML forecasting with Random Forest/XGBoost/LightGBM/time-series models.
6. Confidence and uncertainty reporting.
7. Email/SMS/WhatsApp alerts.
8. Agentic AI for retrieve → validate → analyze → compare → detect → recommend → report → alert.

### Long-term architecture
`Climate + Water + Satellite + GIS → Data Engine → Analytics/ML → Risk Engine → AI Agent → Recommendation/Alert → Decision-maker → Action → Monitoring`

### Scientific roadmap
The major next step is local calibration/validation using long-term observations, laboratory measurements, local standards, hydrological data, satellite observations and expert review. Strong predictive or operational claims should only follow validation.

### Hackathon description
**AquaGuard AI is an AI-powered environmental decision-support platform that integrates climate data, optional water-quality observations and analytical risk indicators to identify potential environmental pressures on water bodies and generate practical recommendations for sustainable water-resource and climate-risk management.**

Relevant SDGs: SDG 6, SDG 13, SDG 14 and, where appropriate, SDG 15.
