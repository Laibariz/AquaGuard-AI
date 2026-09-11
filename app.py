import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from groq import Groq

from workflow import run_workflow
from prompt import SYSTEM_PROMPT, build_report_prompt


st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #071A2B 0%, #0B2538 100%) !important;
            color: #FFFFFF !important;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1250px;
        }

        .hero {
            padding: 28px 30px;
            border-radius: 22px;
            background: linear-gradient(135deg, #0B6E99, #123C69);
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 12px 35px rgba(0,0,0,.18);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            color: #FFFFFF !important;
        }

        .hero p {
            margin: 8px 0 0;
            color: #D9F4FF !important;
            font-size: 1.05rem;
        }

        .metric-card {
            background: #0E2A40;
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #1E425E;
            box-shadow: 0 5px 18px rgba(0,0,0,.2);
            color: #FFFFFF !important;
        }

        .metric-card h2,
        .metric-card h3,
        .metric-card p {
            color: #FFFFFF !important;
        }

        .small-label {
            color: #8DA4B5;
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .risk-high {
            color: #FF6B6B;
            font-weight: 800;
        }

        .risk-medium {
            color: #FFB703;
            font-weight: 800;
        }

        .risk-low {
            color: #38EF7D;
            font-weight: 800;
        }

        .section-title,
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: #EAF6FF !important;
        }

        [data-testid="stSidebar"] {
            background: #051320;
        }

        [data-testid="stSidebar"] * {
            color: #EAF6FF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """
    Read a secret from environment variables first,
    then from Streamlit Secrets.
    """
    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name, default)
        return value if value else default
    except Exception:
        return default


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "temperature_2m_max",
                "precipitation_sum",
                "et0_fao_evapotranspiration",
                "relative_humidity_2m_mean",
                "wind_speed_10m_max",
            ]
        ),
        "timezone": "auto",
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if "daily" not in payload:
        raise ValueError("Open-Meteo returned no daily data.")

    return pd.DataFrame(payload["daily"])


def get_risk_css(risk_level: str) -> str:
    if risk_level == "HIGH":
        return "risk-high"

    if risk_level == "MEDIUM":
        return "risk-medium"

    return "risk-low"


def generate_ai_report(context):
    """
    Generate the environmental intelligence report using
    the centralized prompt.py module.
    """

    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        return (
            "Groq API key not configured. The dashboard is still fully usable. "
            "Add GROQ_API_KEY to your local .env or Streamlit Cloud Secrets "
            "to enable the AI interpretation and recommendations."
        )

    model = get_secret(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )

    client = Groq(api_key=api_key)

    report_prompt = build_report_prompt(context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": report_prompt,
                },
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content

    except Exception as exc:
        return f"AI analysis could not be generated: {exc}"


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>💧 AquaGuard AI</h1>
        <p>AI-Powered Water & Climate Risk Intelligence Platform</p>
        <p>
            Climate signals • Water-quality screening •
            Risk intelligence • Action recommendations
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("🌍 Assessment Setup")

    location_name = st.text_input(
        "Water body / location",
        "Rawal Lake, Islamabad",
    )

    lat = st.number_input(
        "Latitude",
        value=33.6844,
        format="%.5f",
    )

    lon = st.number_input(
        "Longitude",
        value=73.0378,
        format="%.5f",
    )

    today = date.today()

    default_start = max(
        date(2020, 1, 1),
        today - timedelta(days=365 * 5),
    )

    start_date = st.date_input(
        "Historical start",
        default_start,
    )

    end_date = st.date_input(
        "Historical end",
        today - timedelta(days=1),
    )

    st.divider()

    st.caption(
        "Optional: upload a CSV containing water-quality measurements."
    )

    uploaded = st.file_uploader(
        "Water-quality CSV",
        type=["csv"],
        help=(
            "Recommended columns: pH, turbidity, "
            "dissolved_oxygen, nitrate, phosphate."
        ),
    )

    run = st.button(
        "🔎 Run AquaGuard Assessment",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.caption(
        "Data source: Open-Meteo Historical Weather API"
    )

    st.caption(
        "AI layer: Groq API"
    )

    st.caption(
        "Screening scores are for MVP decision support, "
        "not regulatory certification."
    )


# ============================================================
# Main Assessment
# ============================================================

if run:

    # --------------------------------------------------------
    # Validate dates
    # --------------------------------------------------------

    if start_date >= end_date:
        st.error(
            "Historical start date must be earlier than the end date."
        )
        st.stop()

    # --------------------------------------------------------
    # Retrieve weather data
    # --------------------------------------------------------

    with st.spinner(
        "Retrieving climate data and calculating indicators..."
    ):

        try:

            weather = fetch_weather(
                float(lat),
                float(lon),
                start_date.isoformat(),
                end_date.isoformat(),
            )

        except Exception as exc:

            st.error(
                f"Open-Meteo request failed: {exc}"
            )

            st.stop()

    # --------------------------------------------------------
    # Read water-quality CSV
    # --------------------------------------------------------

    wq = None

    if uploaded is not None:

        try:

            wq = pd.read_csv(uploaded)

            wq.columns = (
                wq.columns
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace("-", "_", regex=False)
            )

        except Exception as exc:

            st.warning(
                f"Could not read water-quality CSV: {exc}"
            )

    # ========================================================
    # NEW CENTRALIZED WORKFLOW
    # ========================================================

    try:

        workflow_result = run_workflow(
            weather_df=weather,
            location_name=location_name,
            latitude=float(lat),
            longitude=float(lon),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            water_quality_df=wq,
        )

    except Exception as exc:

        st.error(
            f"AquaGuard workflow failed: {exc}"
        )

        st.stop()

    # --------------------------------------------------------
    # Extract workflow results
    # --------------------------------------------------------

    climate = workflow_result["climate"]

    water_quality = workflow_result["water_quality"]

    overall_data = workflow_result["overall"]

    ai_context = workflow_result["ai_context"]

    # Climate score
    climate_score = climate["climate_score"]

    climate_factors = {
        "Temperature stress": climate["temperature_stress"],
        "Dry-day pressure": climate["dry_day_pressure"],
        "Evapotranspiration pressure": climate["et0_stress"],
    }

    # Water-quality score
    if water_quality is None:

        water_score = None
        water_factors = {}

    else:

        water_score = water_quality["water_quality_score"]

        water_factors = water_quality[
            "pressure_indicators"
        ]

    # Overall score
    overall = overall_data["overall_score"]

    label = overall_data["risk_level"]

    css = get_risk_css(label)

    # ========================================================
    # Summary
    # ========================================================

    st.markdown(
        "### 📊 Environmental Risk Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    # Overall
    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">
                    Overall Risk
                </div>

                <h2>
                    {overall:.0f}/100
                </h2>

                <div class="{css}">
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Climate
    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">
                    Climate Risk
                </div>

                <h2>
                    {climate_score:.0f}/100
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Water quality
    with c3:

        wq_display = (
            "N/A"
            if water_score is None
            else f"{water_score:.0f}/100"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">
                    Water Quality
                </div>

                <h2>
                    {wq_display}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Period
    with c4:

        avg_temp = weather[
            "temperature_2m_mean"
        ].mean()

        rain = weather[
            "precipitation_sum"
        ].sum()

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">
                    Period
                </div>

                <h3>
                    {start_date} → {end_date}
                </h3>

                <p>
                    Mean temp: {avg_temp:.1f}°C
                    <br>
                    Total precipitation: {rain:.1f} mm
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # Climate Indicators
    # ========================================================

    st.markdown(
        "### 🌡️ Climate Indicators"
    )

    chart_df = weather.copy()

    chart_df["time"] = pd.to_datetime(
        chart_df["time"]
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Temperature",
            "Precipitation",
            "Water Stress Signals",
        ]
    )

    # Temperature
    with tab1:

        fig = px.line(
            chart_df,
            x="time",
            y=[
                "temperature_2m_mean",
                "temperature_2m_max",
            ],
            labels={
                "value": "Temperature (°C)",
                "time": "Date",
                "variable": "Series",
            },
            title="Temperature trend",
        )

        fig.update_layout(
            height=380,
            legend_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # Precipitation
    with tab2:

        fig = px.bar(
            chart_df,
            x="time",
            y="precipitation_sum",
            labels={
                "precipitation_sum": "Precipitation (mm)",
                "time": "Date",
            },
            title="Daily precipitation",
        )

        fig.update_layout(
            height=380,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # Water stress
    with tab3:

        stress_cols = [
            "et0_fao_evapotranspiration",
            "relative_humidity_2m_mean",
        ]

        available = [
            col
            for col in stress_cols
            if col in chart_df.columns
        ]

        if available:

            fig = px.line(
                chart_df,
                x="time",
                y=available,
                labels={
                    "value": "Value",
                    "time": "Date",
                    "variable": "Indicator",
                },
                title=(
                    "Evapotranspiration "
                    "and humidity"
                ),
            )

            fig.update_layout(
                height=380,
                legend_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # ========================================================
    # Water Quality
    # ========================================================

    st.markdown(
        "### 🧪 Water-Quality Screening"
    )

    if wq is not None and not wq.empty:

        st.dataframe(
            wq.head(100),
            use_container_width=True,
        )

        if water_factors:

            factor_df = pd.DataFrame(
                {
                    "Indicator": list(
                        water_factors.keys()
                    ),
                    "Pressure score": list(
                        water_factors.values()
                    ),
                }
            )

            fig = px.bar(
                factor_df,
                x="Pressure score",
                y="Indicator",
                orientation="h",
                range_x=[0, 100],
                title=(
                    "Water-quality "
                    "pressure indicators"
                ),
            )

            fig.update_layout(
                height=330,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:

        st.info(
            "No water-quality CSV supplied. "
            "Climate-only screening is being shown."
        )

    # ========================================================
    # Weather summary
    # ========================================================

    weather_summary = {
        "mean_temperature_c": round(
            float(
                weather[
                    "temperature_2m_mean"
                ].mean()
            ),
            2,
        ),

        "max_temperature_c": round(
            float(
                weather[
                    "temperature_2m_max"
                ].max()
            ),
            2,
        ),

        "total_precipitation_mm": round(
            float(
                weather[
                    "precipitation_sum"
                ].sum()
            ),
            2,
        ),

        "mean_et0_mm": round(
            float(
                weather[
                    "et0_fao_evapotranspiration"
                ].mean()
            ),
            2,
        ),

        "mean_relative_humidity_pct": round(
            float(
                weather[
                    "relative_humidity_2m_mean"
                ].mean()
            ),
            2,
        ),
    }

    # Add weather summary to the centralized AI context.
    ai_context["weather_summary"] = weather_summary

    # ========================================================
    # AI Environmental Intelligence
    # ========================================================

    st.markdown(
        "### 🤖 AI Environmental Intelligence"
    )

    with st.spinner(
        "Generating AI interpretation..."
    ):

        report = generate_ai_report(
            ai_context
        )

    st.markdown(report)

    # ========================================================
    # Export
    # ========================================================

    st.markdown(
        "### 📥 Export"
    )

    summary = pd.DataFrame(
        [
            {
                "location": location_name,
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "overall_risk": round(
                    overall,
                    2,
                ),
                "risk_level": label,
                "climate_risk": round(
                    climate_score,
                    2,
                ),
                "water_quality_risk": (
                    None
                    if water_score is None
                    else round(
                        water_score,
                        2,
                    )
                ),
                "mean_temperature_c": round(
                    float(
                        weather[
                            "temperature_2m_mean"
                        ].mean()
                    ),
                    2,
                ),
                "total_precipitation_mm": round(
                    float(
                        weather[
                            "precipitation_sum"
                        ].sum()
                    ),
                    2,
                ),
            }
        ]
    )

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


# ============================================================
# Initial / Empty State
# ============================================================

else:

    st.markdown(
        "### Start an environmental assessment"
    )

    st.write(
        "Enter a water body or location, confirm its coordinates, "
        "optionally upload water-quality measurements, and run "
        "the assessment."
    )

    a, b, c = st.columns(3)

    with a:

        st.info(
            "**1. Climate Data**\n\n"
            "Retrieve historical temperature, precipitation, "
            "evapotranspiration and humidity."
        )

    with b:

        st.info(
            "**2. Risk Analytics**\n\n"
            "Use the centralized AquaGuard workflow to convert "
            "environmental indicators into transparent screening scores."
        )

    with c:

        st.info(
            "**3. AI Intelligence**\n\n"
            "Use the centralized AquaGuard prompts with Groq "
            "to explain findings and propose practical actions."
        )
