# solar_api/main.py

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .analysis.core import generate_perfect_solar_analysis  # Use a relative import

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Solar Analysis API", description="API for solar power generation analysis and prediction")

origins = [
    "http://localhost",
    "http://localhost:3000",  
    "http://127.0.0.1:8000",
]

# --- Add CORS Middleware to the app ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,      
    allow_methods=["*"],         
    allow_headers=["*"],         
)

class SolarInput(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_name: Optional[str] = None
    panel_area: float = 20.0
    current_tilt: float = 35.0
    current_azimuth: float = 180.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@app.get("/")
async def root():
    """
    Root endpoint providing API information and available endpoints.
    """
    return {
        "message": "Solar Analysis API",
        "description": "API for solar power generation analysis and prediction",
        "version": "1.0.0",
        "endpoints": {
            "root": "/",
            "solar_analysis": "/api/solar-analysis",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "usage": {
            "method": "POST",
            "endpoint": "/api/solar-analysis",
            "description": "Submit solar analysis parameters to get a comprehensive report",
        },
    }


@app.post("/api/solar-analysis")
async def get_solar_analysis_report(input_data: SolarInput):
    """
    Runs the full solar analysis and returns the report.
    Accepts lat/lon OR a location name string.
    """
    report = generate_perfect_solar_analysis(
        location_name=input_data.location_name,
        lat=input_data.lat,
        lon=input_data.lon,
        panel_area=input_data.panel_area,
        current_tilt=input_data.current_tilt,
        current_azimuth=input_data.current_azimuth,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
    )

    if report and "error" in report:
        raise HTTPException(status_code=400, detail=report["error"])
    elif report is None:
        raise HTTPException(status_code=500, detail="Analysis failed to complete.")

    return report
