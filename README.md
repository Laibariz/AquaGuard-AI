# AquaGuard AI 💧

**AI-Powered Water & Climate Risk Intelligence Platform**

AquaGuard AI is a Streamlit MVP that combines historical climate/weather indicators, optional water-quality measurements, transparent screening scores, and a Groq-powered generative AI layer to produce environmental risk intelligence and practical recommendations.

## Why this project?

The MVP addresses the gap between raw environmental data and understandable decision support.

It is aligned primarily with:

- **SDG 6 — Clean Water & Sanitation**
- **SDG 13 — Climate Action**
- Potentially **SDG 14 — Life Below Water**, when aquatic ecosystem impacts are supported by the available evidence.

## MVP workflow

1. User enters a water body/location and coordinates.
2. The application retrieves historical weather data from the Open-Meteo Historical Weather API.
3. The application calculates transparent climate-stress indicators.
4. User may upload a CSV containing water-quality observations.
5. The application calculates an indicative water-quality pressure score.
6. Climate and water-quality scores are combined into an overall screening score.
7. Groq generates an AI environmental interpretation and recommendations.
8. User can download a CSV assessment and AI report.

> **Important:** The scoring system is an MVP screening model. It is not a regulatory water-quality certification, medical system, or substitute for field sampling or environmental authority assessment.

## Data sources

### Open-Meteo

The application uses the Open-Meteo Historical Weather API endpoint:

`https://archive-api.open-meteo.com/v1/archive`

The historical API supports latitude/longitude, date ranges and variables including temperature, precipitation, evapotranspiration, humidity and wind. Open-Meteo documents an API key requirement for certain commercial/reserved resources; this MVP uses the public endpoint for non-commercial development.

No Open-Meteo API key is required for this MVP.

### Groq

Groq provides the generative AI layer. The app reads `GROQ_API_KEY` from the environment or Streamlit Secrets.

The default model in this project is:

`openai/gpt-oss-20b`

You can change it using `GROQ_MODEL`.

## Input

### Required

- Water body/location name
- Latitude
- Longitude
- Historical date range

### Optional water-quality CSV

Recommended column names:

- `pH`
- `turbidity`
- `dissolved_oxygen`
- `nitrate`
- `phosphate`

Example:

```csv
pH,turbidity,dissolved_oxygen,nitrate,phosphate
7.8,12,6.2,8,0.5
7.6,10,6.5,7,0.4
7.9,15,5.9,9,0.6
```

The application also normalizes spaces to underscores in uploaded column names.

## Outputs

- Overall environmental screening score
- Climate risk score
- Water-quality pressure score when CSV is provided
- Temperature trend
- Precipitation trend
- Evapotranspiration/humidity signals
- Water-quality indicator chart
- AI environmental assessment
- Recommended actions
- Downloadable CSV assessment
- Downloadable AI report

## Project structure

```text
aquaguard-ai/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Local setup

### 1. Create a project folder

```bash
mkdir aquaguard-ai
cd aquaguard-ai
```

Put the five project files into this folder.

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your local environment file

Copy:

```text
.env.example
```

to:

```text
.env
```

Then put your Groq API key into `.env`:

```text
GROQ_API_KEY=gsk_your_real_key
GROQ_MODEL=openai/gpt-oss-20b
```

The application currently reads Streamlit Secrets/environment variables. If you want `.env` to be loaded automatically during local development, add `from dotenv import load_dotenv` and `load_dotenv()` near the imports in `app.py`.

### 5. Run the app

```bash
streamlit run app.py
```

The terminal will show the local URL, normally:

```text
http://localhost:8501
```

## Getting the Groq API key

1. Create/sign in to a Groq account.
2. Open the Groq Console API Keys area.
3. Create an API key.
4. Copy it immediately and store it in `.env` locally.
5. Never place the real key in `app.py`.
6. Never commit `.env` to GitHub.

Groq's documentation recommends environment variables or secret-management systems for API keys.

## How Open-Meteo and Groq work together

They perform two different jobs.

```text
Open-Meteo
   ↓
Climate/weather observations
   ↓
Python/Pandas analytics
   ↓
Risk indicators + summary
   ↓
Groq API
   ↓
AI interpretation + recommendations
   ↓
Streamlit dashboard
```

Open-Meteo supplies the environmental data. Groq does not replace the environmental data source.

Groq receives a compact structured summary rather than the entire raw dataset. This keeps the AI request smaller and makes the system easier to control.

## GitHub deployment

### 1. Create a GitHub repository

Example:

```text
aquaguard-ai
```

### 2. Upload these files

- `app.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`

Do **not** upload `.env`.

### 3. Commit and push

Using Git:

```bash
git init
git add .
git commit -m "Initial AquaGuard AI MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/aquaguard-ai.git
git push -u origin main
```

### 4. Deploy with Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Connect/authorize your GitHub account.
4. Select **Create app**.
5. Select your `aquaguard-ai` repository.
6. Select branch `main`.
7. Set the entrypoint to `app.py`.
8. Open **Advanced settings**.
9. Select a supported Python version.
10. Add the Streamlit Secrets.

Use:

```toml
GROQ_API_KEY = "gsk_your_real_key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

11. Click **Deploy**.

The application should install the packages from `requirements.txt` and launch.

## Security

Never commit:

```text
.env
.streamlit/secrets.toml
```

Never hard-code:

```python
GROQ_API_KEY = "gsk-..."
```

For Streamlit Community Cloud, put the real secret in the app's Secrets configuration.

## Important scientific limitation

The climate and water-quality scores are deliberately transparent MVP indicators. They should not be presented as official regulatory thresholds or validated predictions.

For a research-grade version, the next stage should include:

- Local Pakistani water-quality datasets
- Validated ecological thresholds
- More robust historical baselines
- Satellite-derived water indicators
- GIS layers
- Statistical validation
- Machine-learning forecasting
- Expert review
- Field validation

## Future roadmap

### Phase 2

- Interactive GIS map
- More water-quality APIs
- Pakistan-specific environmental datasets
- Satellite imagery
- Reservoir/lake boundary detection
- Eutrophication indicators

### Phase 3

- Machine-learning risk forecasting
- Anomaly detection
- Climate scenario analysis
- Automated environmental reports
- Multi-location comparison

### Phase 4

- Agentic AI workflow
- Automated data collection
- Source verification
- Scheduled monitoring
- Alerts for unusual environmental conditions
- Decision-support workflows for NGOs and authorities

## Hackathon positioning

AquaGuard AI demonstrates:

- Generative AI
- Data analytics
- API integration
- Environmental intelligence
- Explainable risk scoring
- Decision support
- SDG alignment
- Streamlit application development
- GitHub-based deployment

**Tagline:**  
**Turn environmental data into actionable water and climate intelligence.**
