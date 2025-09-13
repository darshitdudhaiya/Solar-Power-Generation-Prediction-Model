# solar_api.py
from typing import Optional  # Import Optional for compatibility
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import solar_analysis_core

app = FastAPI(title="Solar Analysis API")


# Define a Pydantic model for request validation
# This model is flexible and accepts either lat/lon or a location_name
class SolarInput(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_name: Optional[str] = None
    panel_area: float = 20.0
    current_tilt: float = 35.0
    current_azimuth: float = 180.0

    # NEW: Add optional date fields with defaults
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@app.post("/api/solar-analysis")
async def get_solar_analysis_report(input_data: SolarInput):
    report = solar_analysis_core.generate_perfect_solar_analysis(
        location_name=input_data.location_name,
        lat=input_data.lat,
        lon=input_data.lon,
        panel_area=input_data.panel_area,
        current_tilt=input_data.current_tilt,
        current_azimuth=input_data.current_azimuth,
        start_date=input_data.start_date,  # Pass new parameters
        end_date=input_data.end_date,  # Pass new parameters
    )

    if report and "error" in report:
        raise HTTPException(status_code=400, detail=report["error"])
    elif report is None:
        raise HTTPException(status_code=500, detail="Analysis failed to complete.")

    return report
