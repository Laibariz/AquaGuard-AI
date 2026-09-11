import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# HTML helper
# ============================================================

def clean_html(html: str) -> str:
    """
    Remove leading whitespace from each line of an HTML/Markdown
    string before passing it to st.markdown().
    """
    return "\n".join(line.strip() for line in html.strip().splitlines())


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
            height: 100%;
        }

        .metric-card h2,
        .metric-card h3,
        .metric-card p {
            color: #FFFFFF !important;
            margin: 4px 0;
        }

        .small-label {
            color: #8DA4B5;
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-weight: 600;
        }

        .card-caption {
            color: #00D2FF !important;
            font-size: .78rem;
            margin-top: 4px !important;
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
    clean_html(
        """
        <div class="hero">
            <h1>💧 AquaGuard AI</h1>
            <p>AI-Powered Water & Climate Risk Intelligence Platform</p>
            <p>
                Climate signals • Water-quality screening •
                Risk intelligence • Action recommendations
            </p>
        </div>
        """
    ),
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

    st.caption("Data source: Open-Meteo Historical Weather API")
    st.caption("AI layer: Groq API")
    st.caption(
        "Screening scores are for MVP decision support, "
        "not regulatory certification."
    )


# ============================================================
# Main Assessment
# ============================================================

if run:

    if start_date >= end_date:
        st.error("Historical start date must be earlier than the end date.")
        st.stop()

    with st.spinner("Retrieving climate data and calculating indicators..."):
        try:
            weather = fetch_weather(
                float(lat),
                float(lon),
                start_date.isoformat(),
                end_date.isoformat(),
            )
        except Exception as exc:
            st.error(f"Open-Meteo request failed: {exc}")
            st.stop()

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
            st.warning(f"Could not read water-quality CSV: {exc}")

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
        st.error(f"AquaGuard workflow failed: {exc}")
        st.stop()

    climate = workflow_result["climate"]
    water_quality = workflow_result["water_quality"]
    overall_data = workflow_result["overall"]
    ai_context = workflow_result["ai_context"]

    climate_score = climate["climate_score"]
    
    if water_quality is None:
        water_score = None
        water_factors = {}
    else:
        water_score = water_quality["water_quality_score"]
        water_factors = water_quality["pressure_indicators"]

    overall = overall_data["overall_score"]
    label = overall_data["risk_level"]
    css = get_risk_css(label)

    # ========================================================
    # Summary Cards (Improved Visual Clarity)
    # ========================================================

    st.markdown("### 📊 Environmental Risk Overview")

    c1, c2, c3, c4 = st.columns(4)

    # Overall Risk
    with c1:
        model_type = "Full Climate & WQ Model" if water_score is not None else "Climate-Only Model"
        st.markdown(
            clean_html(
                f"""
                <div class="metric-card">
                    <div class="small-label">Overall Risk Score</div>
                    <h2>{overall:.0f}<span style="font-size: 1.1rem; color: #8DA4B5;">/100</span></h2>
                    <div class="{css}">{label} RISK</div>
                    <p class="card-caption">ℹ️ {model_type}</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Climate Risk
    with c2:
        st.markdown(
            clean_html(
                f"""
                <div class="metric-card">
                    <div class="small-label">Climate Stress Risk</div>
                    <h2>{climate_score:.0f}<span style="font-size: 1.1rem; color: #8DA4B5;">/100</span></h2>
                    <p class="card-caption">Heat, Drought & Evaporation Pressure</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Water Quality Pressure
    with c3:
        if water_score is None:
            wq_display = "N/A"
            wq_sub = "No CSV uploaded"
        else:
            wq_display = f"{water_score:.0f}<span style='font-size: 1.1rem; color: #8DA4B5;'>/100</span>"
            wq_sub = "Contamination / Quality Deficit Score"

        st.markdown(
            clean_html(
                f"""
                <div class="metric-card">
                    <div class="small-label">Water Quality Pressure</div>
                    <h2>{wq_display}</h2>
                    <p class="card-caption">{wq_sub}</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # Period & Climate Stats
    with c4:
        avg_temp = weather["temperature_2m_mean"].mean()
        rain = weather["precipitation_sum"].sum()

        st.markdown(
            clean_html(
                f"""
                <div class="metric-card">
                    <div class="small-label">Assessment Range</div>
                    <h3>{start_date} → {end_date}</h3>
                    <p style="font-size: 0.85rem; margin-top: 6px;">
                        • Mean temp: <b>{avg_temp:.1f}°C</b><br>
                        • Total rainfall: <b>{rain:.1f} mm</b>
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # ========================================================
    # Climate Indicators (Fixed Plotly Charts)
    # ========================================================

    st.markdown("### 🌡️ Climate Indicators")

    chart_df = weather.copy()
    chart_df["time"] = pd.to_datetime(chart_df["time"])

    tab1, tab2, tab3 = st.tabs(
        [
            "Temperature Trends",
            "Precipitation & Dry Spells",
            "Water Stress Signals (Dual-Axis)",
        ]
    )

    # Tab 1: Temperature
    with tab1:
        fig = px.line(
            chart_df,
            x="time",
            y=["temperature_2m_mean", "temperature_2m_max"],
            labels={
                "value": "Temperature (°C)",
                "time": "Date",
                "variable": "Metric",
            },
            title="Mean vs Maximum Daily Temperature",
            color_discrete_sequence=["#00D2FF", "#FF6B6B"],
        )
        fig.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Precipitation
    with tab2:
        fig = px.bar(
            chart_df,
            x="time",
            y="precipitation_sum",
            labels={"precipitation_sum": "Precipitation (mm)", "time": "Date"},
            title="Daily Rainfall Volume",
            color_discrete_sequence=["#38EF7D"],
        )
        fig.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            yaxis=dict(showgrid=True, gridcolor="#1E425E"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Water Stress Signals (Fixed Dual Axis)
    with tab3:
        if "et0_fao_evapotranspiration" in chart_df.columns and "relative_humidity_2m_mean" in chart_df.columns:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(
                    x=chart_df["time"],
                    y=chart_df["et0_fao_evapotranspiration"],
                    name="Evapotranspiration (mm/day)",
                    line=dict(color="#FFB703", width=2),
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=chart_df["time"],
                    y=chart_df["relative_humidity_2m_mean"],
                    name="Relative Humidity (%)",
                    line=dict(color="#00D2FF", width=1.5, dash="dash"),
                ),
                secondary_y=True,
            )

            fig.update_xaxes(title_text="Date", showgrid=False)
            fig.update_yaxes(
                title_text="Evapotranspiration (mm/day)",
                secondary_y=False,
                showgrid=True,
                gridcolor="#1E425E",
                title_font=dict(color="#FFB703"),
            )
            fig.update_yaxes(
                title_text="Relative Humidity (%)",
                secondary_y=True,
                showgrid=False,
                title_font=dict(color="#00D2FF"),
            )

            fig.update_layout(
                title_text="Evapotranspiration vs. Relative Humidity",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # Water Quality
    # ========================================================

    st.markdown("### 🧪 Water-Quality Screening")

    if wq is not None and not wq.empty:
        st.dataframe(wq.head(100), use_container_width=True)

        if water_factors:
            factor_df = pd.DataFrame(
                {
                    "Indicator": list(water_factors.keys()),
                    "Pressure score": list(water_factors.values()),
                }
            )

            fig = px.bar(
                factor_df,
                x="Pressure score",
                y="Indicator",
                orientation="h",
                range_x=[0, 100],
                title="Water Quality Pressure Breakdown",
                color="Pressure score",
                color_continuous_scale="Reds",
            )
            fig.update_layout(
                height=330,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No water-quality CSV supplied. Assessment based on climate-only risk model.")

    # ========================================================
    # Weather Summary & AI Integration
    # ========================================================

    weather_summary = {
        "mean_temperature_c": round(float(weather["temperature_2m_mean"].mean()), 2),
        "max_temperature_c": round(float(weather["temperature_2m_max"].max()), 2),
        "total_precipitation_mm": round(float(weather["precipitation_sum"].sum()), 2),
        "mean_et0_mm": round(float(weather["et0_fao_evapotranspiration"].mean()), 2),
        "mean_relative_humidity_pct": round(float(weather["relative_humidity_2m_mean"].mean()), 2),
    }

    ai_context["weather_summary"] = weather_summary

    st.markdown("### 🤖 AI Environmental Intelligence")

    with st.spinner("Generating AI interpretation..."):
        report = generate_ai_report(ai_context)

    st.markdown(report)

    # ========================================================
    # Export
    # ========================================================

    st.markdown("### 📥 Export")

    summary = pd.DataFrame(
        [
            {
                "location": location_name,
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "overall_risk": round(overall, 2),
                "risk_level": label,
                "climate_risk": round(climate_score, 2),
                "water_quality_pressure": None if water_score is None else round(water_score, 2),
                "mean_temperature_c": weather_summary["mean_temperature_c"],
                "total_precipitation_mm": weather_summary["total_precipitation_mm"],
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
    st.markdown("### Start an environmental assessment")
    st.write(
        "Enter a water body or location, confirm its coordinates, "
        "optionally upload water-quality measurements, and run the assessment."
    )

    a, b, c = st.columns(3)

    with a:
        st.info(
            "**1. Climate Data**\n\n"
            "Retrieve historical temperature, precipitation, "
            "evapotranspiration, and humidity."
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

