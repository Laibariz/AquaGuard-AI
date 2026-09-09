import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ---------- Styling ----------
st.markdown("""
<style>
    /* Full dark theme background across the entire page */
    .stApp {
        background: linear-gradient(180deg, #071A2B 0%, #0B2538 100%) !important;
        color: #FFFFFF !important;
    }
    .block-container { padding-top: 1.5rem; max-width: 1250px; }
    
    /* Hero Banner */
    .hero {
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(135deg, #0B6E99, #123C69);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
    }
    .hero h1 { margin: 0; font-size: 2.5rem; color: #FFFFFF !important; }
    .hero p { margin: 8px 0 0; color: #D9F4FF !important; font-size: 1.05rem; }
    
    /* Dark-mode compatible metric cards */
    .metric-card {
        background: #0E2A40; 
        border-radius: 16px; 
        padding: 18px;
        border: 1px solid #1E425E; 
        box-shadow: 0 5px 18px rgba(0,0,0,.2);
        color: #FFFFFF !important;
    }
    .metric-card h2, .metric-card h3, .metric-card p {
        color: #FFFFFF !important;
    }
    .small-label { color: #8DA4B5; font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; }
    
    /* Risk Levels */
    .risk-high { color: #FF6B6B; font-weight: 800; }
    .risk-medium { color: #FFB703; font-weight: 800; }
    .risk-low { color: #38EF7D; font-weight: 800; }
    
    /* Section Titles & Global Text Fixes */
    .section-title, h1, h2, h3, h4, h5, h6, p, label, span { 
        color: #EAF6FF !important; 
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background: #051320; }
    [data-testid="stSidebar"] * { color: #EAF6FF !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
        return value if value else default
    except Exception:
        return default


def clamp(value, low=0, high=100):
    return float(max(low, min(high, value)))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
        ]),
        "timezone": "auto",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "daily" not in payload:
        raise ValueError("Open-Meteo returned no daily data.")
    return pd.DataFrame(payload["daily"])


def calculate_climate_risk(df):
    d = df.copy()
    # These are transparent MVP indicators, not regulatory thresholds.
    temp = d["temperature_2m_mean"].dropna()
    precip = d["precipitation_sum"].dropna()
    et0 = d["et0_fao_evapotranspiration"].dropna()

    temp_stress = clamp((temp.mean() - 20) / 18 * 100) if len(temp) else 0
    dry_days = (precip < 1).mean() * 100 if len(precip) else 0
    et0_stress = clamp(et0.mean() / 7 * 100) if len(et0) else 0

    score = 0.40 * temp_stress + 0.35 * dry_days + 0.25 * et0_stress
    return clamp(score), {
        "Temperature stress": temp_stress,
        "Dry-day pressure": dry_days,
        "Evapotranspiration pressure": et0_stress,
    }


def water_quality_risk(wq):
    if wq is None or wq.empty:
        return None, {}

    scores = {}
    if "pH" in wq.columns:
        v = pd.to_numeric(wq["pH"], errors="coerce").dropna()
        if len(v):
            # Indicative freshwater comfort band for an MVP; not a legal standard.
            scores["pH pressure"] = clamp(np.mean(np.maximum(0, np.abs(v - 7.5) - 1.0)) * 35)

    if "turbidity" in wq.columns:
        v = pd.to_numeric(wq["turbidity"], errors="coerce").dropna()
        if len(v):
            scores["Turbidity pressure"] = clamp(v.mean() / 20 * 100)

    if "dissolved_oxygen" in wq.columns:
        v = pd.to_numeric(wq["dissolved_oxygen"], errors="coerce").dropna()
        if len(v):
            scores["Low dissolved oxygen pressure"] = clamp((7 - v.mean()) / 7 * 100)

    if "nitrate" in wq.columns:
        v = pd.to_numeric(wq["nitrate"], errors="coerce").dropna()
        if len(v):
            scores["Nitrate pressure"] = clamp(v.mean() / 10 * 100)

    if "phosphate" in wq.columns:
        v = pd.to_numeric(wq["phosphate"], errors="coerce").dropna()
        if len(v):
            scores["Phosphate pressure"] = clamp(v.mean() / 1 * 100)

    if not scores:
        return None, {}

    return clamp(np.mean(list(scores.values()))), scores


def risk_label(score):
    if score >= 67:
        return "HIGH", "risk-high"
    if score >= 34:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"


def generate_ai_report(location, lat, lon, climate_score, climate_factors,
                       water_score, water_factors, weather_summary):
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        return (
            "Groq API key not configured. The dashboard is still fully usable. "
            "Add GROQ_API_KEY to your local .env or Streamlit Cloud Secrets to enable "
            "the AI interpretation and recommendations."
        )

    model = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")
    client = Groq(api_key=api_key)

    overall = climate_score if water_score is None else (0.55 * climate_score + 0.45 * water_score)
    label, _ = risk_label(overall)

    prompt = f"""
You are AquaGuard AI, an environmental decision-support assistant.
Analyze the structured environmental indicators below.

Important:
- This is an MVP screening tool, not a regulatory water-quality assessment.
- Do not invent measurements or claim certainty.
- Clearly distinguish observations from possible explanations.
- Give practical, non-alarmist recommendations.
- Keep the report concise and suitable for researchers, NGOs, students and local decision-makers.

Location: {location}
Coordinates: {lat}, {lon}
Overall screening risk: {overall:.1f}/100 ({label})
Climate risk: {climate_score:.1f}/100
Climate factors: {climate_factors}
Water-quality risk: {water_score if water_score is not None else "not provided"}
Water-quality factors: {water_factors}
Weather summary: {weather_summary}

Return exactly these sections:
1. Executive Assessment
2. Key Risk Drivers
3. Environmental Concerns
4. Recommended Actions
5. SDG Relevance

Mention SDG 6 and SDG 13 where relevant, and SDG 14 only if aquatic ecosystem impacts are reasonably supported.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful environmental analytics assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"AI analysis could not be generated: {exc}"


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>💧 AquaGuard AI</h1>
    <p>AI-Powered Water & Climate Risk Intelligence Platform</p>
    <p>Climate signals • Water-quality screening • Risk intelligence • Action recommendations</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🌍 Assessment Setup")
    location_name = st.text_input("Water body / location", "Rawal Lake, Islamabad")
    lat = st.number_input("Latitude", value=33.6844, format="%.5f")
    lon = st.number_input("Longitude", value=73.0378, format="%.5f")

    today = date.today()
    default_start = max(date(2020, 1, 1), today - timedelta(days=365 * 5))
    start_date = st.date_input("Historical start", default_start)
    end_date = st.date_input("Historical end", today - timedelta(days=1))

    st.divider()
    st.caption("Optional: upload a CSV containing water-quality measurements.")
    uploaded = st.file_uploader(
        "Water-quality CSV",
        type=["csv"],
        help="Recommended columns: pH, turbidity, dissolved_oxygen, nitrate, phosphate."
    )

    run = st.button("🔎 Run AquaGuard Assessment", use_container_width=True, type="primary")

    st.divider()
    st.caption("Data source: Open-Meteo Historical Weather API")
    st.caption("AI layer: Groq API")
    st.caption("Screening scores are for MVP decision support, not regulatory certification.")

# ---------- Main ----------
if run:
    if start_date >= end_date:
        st.error("Historical start date must be earlier than the end date.")
        st.stop()

    with st.spinner("Retrieving climate data and calculating indicators..."):
        try:
            weather = fetch_weather(
                float(lat), float(lon),
                start_date.isoformat(), end_date.isoformat()
            )
        except Exception as exc:
            st.error(f"Open-Meteo request failed: {exc}")
            st.stop()

    wq = None
    if uploaded is not None:
        try:
            wq = pd.read_csv(uploaded)
            wq.columns = [c.strip().lower().replace(" ", "_") for c in wq.columns]
        except Exception as exc:
            st.warning(f"Could not read water-quality CSV: {exc}")

    climate_score, climate_factors = calculate_climate_risk(weather)
    water_score, water_factors = water_quality_risk(wq)

    overall = climate_score if water_score is None else (0.55 * climate_score + 0.45 * water_score)
    label, css = risk_label(overall)

    # Summary
    st.markdown("### 📊 Environmental Risk Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="small-label">Overall Risk</div><h2>{overall:.0f}/100</h2><div class="{css}">{label}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="small-label">Climate Risk</div><h2>{climate_score:.0f}/100</h2></div>', unsafe_allow_html=True)
    with c3:
        wq_display = "N/A" if water_score is None else f"{water_score:.0f}/100"
        st.markdown(f'<div class="metric-card"><div class="small-label">Water Quality</div><h2>{wq_display}</h2></div>', unsafe_allow_html=True)
    with c4:
        avg_temp = weather["temperature_2m_mean"].mean()
        rain = weather["precipitation_sum"].sum()
        st.markdown(f'<div class="metric-card"><div class="small-label">Period</div><h3>{start_date} → {end_date}</h3><p>Mean temp: {avg_temp:.1f}°C<br>Total precipitation: {rain:.1f} mm</p></div>', unsafe_allow_html=True)

    st.markdown("### 🌡️ Climate Indicators")
    chart_df = weather.copy()
    chart_df["time"] = pd.to_datetime(chart_df["time"])

    tab1, tab2, tab3 = st.tabs(["Temperature", "Precipitation", "Water Stress Signals"])
    with tab1:
        fig = px.line(
            chart_df, x="time", y=["temperature_2m_mean", "temperature_2m_max"],
            labels={"value": "Temperature (°C)", "time": "Date", "variable": "Series"},
            title="Temperature trend"
        )
        fig.update_layout(height=380, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            chart_df, x="time", y="precipitation_sum",
            labels={"precipitation_sum": "Precipitation (mm)", "time": "Date"},
            title="Daily precipitation"
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        stress_cols = ["et0_fao_evapotranspiration", "relative_humidity_2m_mean"]
        available = [c for c in stress_cols if c in chart_df.columns]
        if available:
            fig = px.line(
                chart_df, x="time", y=available,
                labels={"value": "Value", "time": "Date", "variable": "Indicator"},
                title="Evapotranspiration and humidity"
            )
            fig.update_layout(height=380, legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧪 Water-Quality Screening")
    if wq is not None and not wq.empty:
        st.dataframe(wq.head(100), use_container_width=True)
        if water_factors:
            factor_df = pd.DataFrame(
                {"Indicator": list(water_factors.keys()), "Pressure score": list(water_factors.values())}
            )
            fig = px.bar(
                factor_df, x="Pressure score", y="Indicator", orientation="h",
                range_x=[0, 100], title="Water-quality pressure indicators"
            )
            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No water-quality CSV supplied. Climate-only screening is being shown.")

    weather_summary = {
        "mean_temperature_c": round(float(weather["temperature_2m_mean"].mean()), 2),
        "max_temperature_c": round(float(weather["temperature_2m_max"].max()), 2),
        "total_precipitation_mm": round(float(weather["precipitation_sum"].sum()), 2),
        "mean_et0_mm": round(float(weather["et0_fao_evapotranspiration"].mean()), 2),
        "mean_relative_humidity_pct": round(float(weather["relative_humidity_2m_mean"].mean()), 2),
    }

    st.markdown("### 🤖 AI Environmental Intelligence")
    with st.spinner("Generating AI interpretation..."):
        report = generate_ai_report(
            location_name, lat, lon, climate_score, climate_factors,
            water_score, water_factors, weather_summary
        )
    st.markdown(report)

    # Downloadable data/report
    st.markdown("### 📥 Export")
    summary = pd.DataFrame([{
        "location": location_name,
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "overall_risk": round(overall, 2),
        "risk_level": label,
        "climate_risk": round(climate_score, 2),
        "water_quality_risk": None if water_score is None else round(water_score, 2),
        "mean_temperature_c": round(float(weather["temperature_2m_mean"].mean()), 2),
        "total_precipitation_mm": round(float(weather["precipitation_sum"].sum()), 2),
    }])

    left, right = st.columns(2)
    with left:
        st.download_button(
            "⬇️ Download assessment CSV",
            summary.to_csv(index=False),
            file_name="aquaguard_assessment.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "⬇️ Download AI report",
            report,
            file_name="aquaguard_ai_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

else:
    st.markdown("### Start an environmental assessment")
    st.write(
        "Enter a water body or location, confirm its coordinates, optionally upload water-quality "
        "measurements, and run the assessment."
    )
    a, b, c = st.columns(3)
    with a:
        st.info("**1. Climate Data**\n\nRetrieve historical temperature, precipitation, evapotranspiration and humidity.")
    with b:
        st.info("**2. Risk Analytics**\n\nConvert environmental indicators into transparent screening scores.")
    with c:
        st.info("**3. AI Intelligence**\n\nUse Groq to explain the findings and propose practical actions.")
