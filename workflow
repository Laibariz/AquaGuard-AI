# AquaGuard AI — reusable workflow module
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return float(np.clip(value, low, high))

def classify_risk(score: float) -> str:
    return 'HIGH' if score >= 67 else ('MEDIUM' if score >= 34 else 'LOW')

def calculate_climate_indicators(weather_df: pd.DataFrame) -> Dict[str, float]:
    if weather_df.empty: raise ValueError('Weather dataset is empty.')
    temp = pd.to_numeric(weather_df['temperature_2m_mean'], errors='coerce')
    precip = pd.to_numeric(weather_df['precipitation_sum'], errors='coerce')
    et0 = pd.to_numeric(weather_df['et0_fao_evapotranspiration'], errors='coerce')
    mean_temp, dry_ratio, mean_et0 = float(temp.mean()), float((precip < 1).mean()), float(et0.mean())
    t = clamp((mean_temp - 20) / 18 * 100)
    d = clamp(dry_ratio * 100)
    e = clamp(mean_et0 / 7 * 100)
    return {'mean_temperature': mean_temp, 'dry_day_ratio': dry_ratio, 'dry_day_pressure': d, 'mean_et0': mean_et0, 'temperature_stress': t, 'et0_stress': e, 'climate_score': clamp(.40*t + .35*d + .25*e)}

def calculate_water_quality_indicators(wq: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if wq is None or wq.empty: return None
    wq = wq.copy(); wq.columns = wq.columns.astype(str).str.strip().str.lower().str.replace(' ', '_', regex=False).str.replace('-', '_', regex=False)
    p = {}
    if 'ph' in wq: p['pH pressure'] = clamp((((pd.to_numeric(wq['ph'], errors='coerce').dropna()-7.5).abs()-1).clip(lower=0)*100).mean())
    if 'turbidity' in wq: p['Turbidity pressure'] = clamp(pd.to_numeric(wq['turbidity'], errors='coerce').dropna().mean()/20*100)
    if 'dissolved_oxygen' in wq: p['Low dissolved oxygen pressure'] = clamp((7-pd.to_numeric(wq['dissolved_oxygen'], errors='coerce').dropna().mean())/7*100)
    if 'nitrate' in wq: p['Nitrate pressure'] = clamp(pd.to_numeric(wq['nitrate'], errors='coerce').dropna().mean()/10*100)
    if 'phosphate' in wq: p['Phosphate pressure'] = clamp(pd.to_numeric(wq['phosphate'], errors='coerce').dropna().mean()/1*100)
    if not p: return {'water_quality_score': None, 'pressure_indicators': {}, 'available_indicators': []}
    return {'water_quality_score': clamp(float(np.mean(list(p.values())))), 'pressure_indicators': p, 'available_indicators': list(p)}

def calculate_overall_risk(climate_score: float, water_quality_score: Optional[float] = None) -> Dict[str, Any]:
    overall = climate_score if water_quality_score is None else .55*climate_score + .45*water_quality_score
    return {'overall_score': clamp(overall), 'risk_level': classify_risk(overall), 'basis': 'Climate indicators only' if water_quality_score is None else 'Climate + water-quality indicators'}

def build_ai_context(location_name, latitude, longitude, start_date, end_date, climate, water_quality, overall):
    return {'location': {'name': location_name, 'latitude': latitude, 'longitude': longitude}, 'analysis_period': {'start': start_date, 'end': end_date}, 'climate': climate, 'water_quality': water_quality, 'overall_risk': overall, 'interpretation_note': 'MVP screening indicators, not regulatory classifications or validated probabilities.'}

def run_workflow(weather_df, location_name, latitude, longitude, start_date, end_date, water_quality_df=None):
    climate = calculate_climate_indicators(weather_df); wq = calculate_water_quality_indicators(water_quality_df)
    overall = calculate_overall_risk(climate['climate_score'], None if wq is None else wq['water_quality_score'])
    return {'climate': climate, 'water_quality': wq, 'overall': overall, 'ai_context': build_ai_context(location_name, latitude, longitude, start_date, end_date, climate, wq, overall)}
