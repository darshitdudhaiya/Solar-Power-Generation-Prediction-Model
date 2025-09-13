# solar_analysis_core.py

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    Timeout,
    RequestException,
)
from urllib3.util.retry import Retry
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
import time
import logging

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime, timedelta

# Define global constants for solar panel calculations
PANEL_AREA = 20.0  # Default panel area in m²
PANEL_EFFICIENCY = 0.20  # Typical panel efficiency (e.g., 20%)
SYSTEM_EFFICIENCY = 0.95  # Typical inverter/system efficiency (e.g., 95%)
ELECTRICITY_RATE = 0.15  # Average electricity rate in $/kWh
INSTALLATION_COST = 5000.0  # Placeholder for typical installation cost

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_json_output(data):
    """
    Recursively replaces NaN, Inf, and -Inf values with None
    to make the output JSON compliant.
    """
    if isinstance(data, dict):
        return {k: clean_json_output(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_output(item) for item in data]
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    elif isinstance(data, np.generic) and (np.isnan(data) or np.isinf(data)):
        return None
    else:
        return data


def create_robust_session():
    """Create a robust HTTP session with retry logic and proper configuration"""
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=2,
        raise_on_redirect=False,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": "Solar Analysis Tool/1.0 (Python requests)",
            "Accept": "application/json",
            "Connection": "close",
        }
    )

    return session


def make_robust_request(url, params=None, max_retries=3, base_delay=1):
    """Make a robust HTTP request with exponential backoff and proper error handling"""
    session = create_robust_session()
    try:
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Making request to {url} (attempt {attempt + 1}/{max_retries})"
                )
                response = session.get(url, params=params, timeout=(10, 60), stream=False)
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as e:
                    raise RequestException(f"Invalid JSON response: {e}")
                logger.info(f"Successfully received response from {url}")
                return data
            except ChunkedEncodingError as e:
                logger.warning(f"Chunked encoding error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
            except (ConnectionError, Timeout) as e:
                logger.warning(f"Connection/timeout error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"HTTP {e.response.status_code} error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise
            except RequestException as e:
                logger.warning(f"Request exception on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
    finally:
        session.close()


def get_coordinates_from_location(location_name):
    """Converts a location name string to latitude and longitude."""
    geolocator = Nominatim(user_agent="solar_analysis_app")
    try:
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            logger.info(f"Successfully geocoded '{location_name}' to ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        else:
            logger.error(f"Could not find coordinates for '{location_name}'.")
            return None, None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.error(f"Geocoding service error for '{location_name}': {e}")
        return None, None


def get_chunked_historical_data(lat, lon, start_date_str, end_date_str, tilt, azimuth):
    """
    Fetches historical data in monthly chunks to avoid timeouts.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    combined_data = {'hourly': {}}

    current_chunk_start = start_date
    while current_chunk_start <= end_date:
        current_chunk_end = (current_chunk_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        
        if current_chunk_end > end_date:
            current_chunk_end = end_date

        chunk_start_str = current_chunk_start.strftime("%Y-%m-%d")
        chunk_end_str = current_chunk_end.strftime("%Y-%m-%d")

        params = {
            "latitude": lat, "longitude": lon, "start_date": chunk_start_str, "end_date": chunk_end_str,
            "hourly": ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "global_tilted_irradiance",
                       "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "temperature_2m",
                       "relative_humidity_2m", "wind_speed_10m", "precipitation", "snow_depth", "sunshine_duration",
                       "pressure_msl", "vapour_pressure_deficit"],
            "daily": ["sunrise", "sunset", "daylight_duration", "sunshine_duration", "shortwave_radiation_sum"],
            "tilt": tilt, "azimuth": azimuth, "timezone": "auto",
        }
        
        try:
            chunk_data = make_robust_request("https://archive-api.open-meteo.com/v1/archive", params=params)
            
            if chunk_data and 'hourly' in chunk_data and not combined_data['hourly']:
                combined_data = chunk_data
            elif chunk_data and 'hourly' in chunk_data:
                for key, values in chunk_data['hourly'].items():
                    combined_data['hourly'][key].extend(values)
            else:
                logger.warning(f"No data returned for chunk {chunk_start_str} to {chunk_end_str}. Skipping.")

        except RequestException as e:
            logger.error(f"Failed to fetch historical data for chunk {chunk_start_str} to {chunk_end_str}: {e}")
            raise

        current_chunk_start = current_chunk_end + timedelta(days=1)
    
    return combined_data


def get_complete_solar_data(lat, lon, start_date, end_date, tilt=35, azimuth=180):
    """
    Wrapper function that calls the chunked data fetching.
    """
    return get_chunked_historical_data(lat, lon, start_date, end_date, tilt, azimuth)


def get_forecast_data(lat, lon, tilt=35, azimuth=180):
    """Fetches the next 7 days of solar-relevant data from Open-Meteo's forecast API."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "global_tilted_irradiance",
                   "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "temperature_2m",
                   "relative_humidity_2m", "wind_speed_10m", "precipitation", "snow_depth", "sunshine_duration",
                   "pressure_msl", "vapour_pressure_deficit"],
        "daily": ["sunrise", "sunset", "daylight_duration", "sunshine_duration", "shortwave_radiation_sum"],
        "tilt": tilt, "azimuth": azimuth, "timezone": "auto",
    }
    return make_robust_request("https://api.open-meteo.com/v1/forecast", params=params)


def get_smart_solar_data(lat, lon, start_date_str, end_date_str, tilt, azimuth):
    """
    Fetches data by smartly splitting the request between the historical and
    forecast APIs based on the current date.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    today = datetime.now().date()
    
    historical_data = {'hourly': {}}
    forecast_data = {}
    
    if end_date.date() < today:
        historical_data = get_chunked_historical_data(lat, lon, start_date_str, end_date_str, tilt, azimuth)
    
    elif start_date.date() > today:
        forecast_response = get_forecast_data(lat, lon, tilt, azimuth)
        
        forecast_df = pd.DataFrame(forecast_response['hourly'])
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        
        filtered_forecast_df = forecast_df[
            (forecast_df['time'].dt.date >= start_date.date()) & 
            (forecast_df['time'].dt.date <= end_date.date())
        ]
        
        forecast_data = {
            'hourly': filtered_forecast_df.to_dict(orient='list'),
            'daily': forecast_response['daily'] 
        }

    else:
        historical_end_date = today - timedelta(days=1)
        historical_end_date_str = historical_end_date.strftime("%Y-%m-%d")
        historical_data = get_chunked_historical_data(lat, lon, start_date_str, historical_end_date_str, tilt, azimuth)
        
        forecast_response = get_forecast_data(lat, lon, tilt, azimuth)
        forecast_df = pd.DataFrame(forecast_response['hourly'])
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        
        filtered_forecast_df = forecast_df[
            (forecast_df['time'].dt.date >= today) & 
            (forecast_df['time'].dt.date <= end_date.date())
        ]
        
        forecast_data = {
            'hourly': filtered_forecast_df.to_dict(orient='list'),
            'daily': forecast_response['daily']
        }
    
    merged_data = historical_data.copy() if historical_data else {'hourly': {}}
    if forecast_data:
        if 'hourly' in merged_data:
            for key, values in forecast_data['hourly'].items():
                if key in merged_data['hourly']:
                    merged_data['hourly'][key].extend(values)
                else:
                    merged_data['hourly'][key] = values
        else:
            merged_data.update(forecast_data)
        
    return merged_data


def calculate_perfect_solar_features(data, lat, lon):
    """Transform raw weather data into solar power predictions"""
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["theoretical_panel_output"] = (df["global_tilted_irradiance"] * PANEL_AREA * PANEL_EFFICIENCY * SYSTEM_EFFICIENCY) / 1000
    df["temp_coefficient"] = 1 + (df["temperature_2m"] - 25) * -0.004
    df["temperature_adjusted_output"] = df["theoretical_panel_output"] * df["temp_coefficient"]
    df["cloud_reduction_factor"] = (df["cloud_cover_low"] * 0.008 + df["cloud_cover_mid"] * 0.005 + df["cloud_cover_high"] * 0.002)
    df["cloud_adjusted_output"] = df["temperature_adjusted_output"] * (1 - df["cloud_reduction_factor"])
    df["snow_blockage"] = np.where(df["snow_depth"] > 0.01, 0.1, 1.0)
    df["rain_cleaning_bonus"] = np.where(df["precipitation"] > 1, 1.02, 1.0)
    df["wind_cooling_factor"] = 1 + (df["wind_speed_10m"] * 0.001)
    df["atmospheric_transmission"] = 1 - (df["relative_humidity_2m"] / 100) * 0.1
    df["predicted_solar_output_kwh"] = (df["cloud_adjusted_output"] * df["snow_blockage"] * df["rain_cleaning_bonus"] * df["wind_cooling_factor"] * df["atmospheric_transmission"])
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["is_summer"] = df["month"].isin([6, 7, 8])
    df["is_peak_sun"] = (df["hour"] >= 10) & (df["hour"] <= 14)
    df["sun_elevation"] = calculate_sun_elevation(df["time"], lat, lon)
    df["air_mass"] = 1 / np.sin(np.radians(df["sun_elevation"]))
    return df


def calculate_sun_elevation(timestamps, lat, lon):
    """Calculate sun elevation angle for each timestamp"""
    elevations = []
    for ts in timestamps:
        day_of_year = ts.dayofyear
        hour_angle = 15 * (ts.hour - 12)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        elevation = np.arcsin(
            np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        elevations.append(np.degrees(elevation))
    return elevations


class PerfectSolarPredictor:
    def __init__(self):
        self.physics_model = self.create_physics_model()
        self.weather_model = SGDRegressor()
        self.temporal_model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5)
        self.solar_model = GradientBoostingRegressor(n_estimators=250, max_depth=6, learning_rate=0.12)
        self.meta_model = LinearRegression()

    def create_physics_model(self):
        def physics_predict(X):
            return (X["global_tilted_irradiance"] * PANEL_AREA * PANEL_EFFICIENCY * SYSTEM_EFFICIENCY / 1000 *
                    (1 + (X["temperature_2m"] - 25) * -0.004))
        return physics_predict

    def prepare_features(self, df):
        weather_features = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "snow_depth", "pressure_msl"]
        temporal_features = ["hour", "day_of_year", "month", "is_summer", "is_peak_sun", "sun_elevation", "air_mass"]
        solar_features = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "global_tilted_irradiance", "sunshine_duration"]
        all_features = weather_features + temporal_features + solar_features
        return {"weather": df[weather_features], "temporal": df[temporal_features], "solar": df[solar_features], "all": df[all_features]}

    def train(self, df, target_column="predicted_solar_output_kwh"):
        features = self.prepare_features(df)
        y = df[target_column]
        self.weather_model.fit(features["weather"], y)
        self.temporal_model.fit(features["temporal"], y)
        self.solar_model.fit(features["solar"], y)
        weather_pred = self.weather_model.predict(features["weather"])
        temporal_pred = self.temporal_model.predict(features["temporal"])
        solar_pred = self.solar_model.predict(features["solar"])
        physics_pred = df.apply(self.physics_model, axis=1)
        meta_features = np.column_stack([weather_pred, temporal_pred, solar_pred, physics_pred])
        self.meta_model.fit(meta_features, y)

    def predict(self, df):
        features = self.prepare_features(df)
        weather_pred = self.weather_model.predict(features["weather"])
        temporal_pred = self.temporal_model.predict(features["temporal"])
        solar_pred = self.solar_model.predict(features["solar"])
        physics_pred = df.apply(self.physics_model, axis=1)
        meta_features = np.column_stack([weather_pred, temporal_pred, solar_pred, physics_pred])
        final_prediction = self.meta_model.predict(meta_features)
        return {"final_prediction": final_prediction, "weather_component": weather_pred, "temporal_component": temporal_pred, "solar_component": solar_pred, "physics_component": physics_pred}


def find_perfect_panel_configuration(df_for_optimization, lat, lon):
    """Finds the optimal panel configuration by calculating tilted irradiance locally."""
    logger.info("Starting local optimization for panel configuration...")
    # The function now takes a DataFrame as input
    df = df_for_optimization.copy()
    optimization_results = []
    # All optimization calculations are done on the single DataFrame
    for tilt in range(0, 91, 5):
        for azimuth in range(-180, 181, 15):
            tilt_rad = np.radians(tilt)
            azimuth_rad = np.radians(azimuth)
            df["local_tilted_irradiance"] = (df["direct_radiation"] * np.sin(np.radians(df["sun_elevation"]) + tilt_rad) +
                                             df["diffuse_radiation"] * (1 - np.cos(tilt_rad)) / 2 +
                                             df["shortwave_radiation"] * 0.2 * (1 - np.cos(tilt_rad)) / 2)
            df["local_tilted_irradiance"] = df["local_tilted_irradiance"].clip(lower=0)
            annual_output = (df["local_tilted_irradiance"].sum() * PANEL_AREA * PANEL_EFFICIENCY / 1000)
            df["month"] = df["time"].dt.month
            summer_output = df[df["month"].isin([6, 7, 8])]["local_tilted_irradiance"].sum()
            winter_output = df[df["month"].isin([12, 1, 2])]["local_tilted_irradiance"].sum()
            optimization_results.append({
                "tilt": tilt, "azimuth": azimuth, "annual_output_kwh": annual_output,
                "summer_output": summer_output, "winter_output": winter_output,
                "seasonal_balance": min(summer_output, winter_output) / max(summer_output, winter_output),
            })
    best_annual = max(optimization_results, key=lambda x: x["annual_output_kwh"])
    best_balanced = max(optimization_results, key=lambda x: x["seasonal_balance"])
    logger.info("Optimization complete.")
    return {"optimal_for_maximum_output": best_annual, "optimal_for_seasonal_balance": best_balanced}


def analyze_configuration_impact(df_for_impact, lat, lon, current_tilt, current_azimuth):
    """
    Compare current setup vs optimal setup with EXACT numbers
    The function now takes a DataFrame as input to avoid redundant API calls.
    """
    # Use the provided DataFrame instead of making a new API call
    current_output = (
        sum(df_for_impact["global_tilted_irradiance"]) * PANEL_AREA * PANEL_EFFICIENCY / 1000
    )
    # Pass the DataFrame to the optimization function to reuse the data
    optimal = find_perfect_panel_configuration(df_for_impact, lat, lon)["optimal_for_maximum_output"]
    optimal_output = optimal["annual_output_kwh"]
    improvement_kwh = optimal_output - current_output
    improvement_percentage = (improvement_kwh / current_output) * 100 if current_output > 0 else 0
    annual_savings = improvement_kwh * ELECTRICITY_RATE
    return {
        "current_annual_output": current_output,
        "optimal_annual_output": optimal_output,
        "improvement_kwh": improvement_kwh,
        "improvement_percentage": improvement_percentage,
        "annual_financial_benefit": annual_savings,
        "optimal_tilt": optimal["tilt"],
        "optimal_azimuth": optimal["azimuth"],
        "payback_period_years": (INSTALLATION_COST / annual_savings if annual_savings > 0 else float("inf")),
    }


def calculate_weather_impact(df, weather_var):
    """Calculate how different weather conditions affect solar output"""
    return {
        "correlation": df[weather_var].corr(df["predicted_solar_output_kwh"]),
        "high_impact_threshold": df[weather_var].quantile(0.9),
        "low_impact_threshold": df[weather_var].quantile(0.1),
        "average_reduction_high": df[df[weather_var] > df[weather_var].quantile(0.9)]["predicted_solar_output_kwh"].mean(),
        "average_reduction_low": df[df[weather_var] < df[weather_var].quantile(0.1)]["predicted_solar_output_kwh"].mean(),
    }


def calculate_solar_potential_rating(df):
    """Calculate overall solar potential rating for the location"""
    annual_output = df["predicted_solar_output_kwh"].sum()
    max_theoretical_output = PANEL_AREA * PANEL_EFFICIENCY * 1000 * 8760
    capacity_factor = (annual_output / max_theoretical_output) * 100
    if capacity_factor > 25: return "Excellent"
    elif capacity_factor > 20: return "Very Good"
    elif capacity_factor > 15: return "Good"
    elif capacity_factor > 10: return "Fair"
    else: return "Poor"


def generate_maintenance_recommendations(df):
    """Generate maintenance schedule based on weather patterns"""
    return {
        "panel_cleaning": {
            "frequency": "Monthly", "priority_months": df.groupby("month")["precipitation"].mean().nsmallest(3).index.tolist(),
            "reason": "Low precipitation months require more frequent cleaning",
        },
        "inspection_schedule": {
            "quarterly_check": "Visual inspection and performance monitoring",
            "annual_check": "Professional inspection and electrical testing",
            "weather_related": "Check after severe weather events",
        },
        "seasonal_tasks": {
            "winter": "Snow removal and ice prevention", "spring": "Post-winter damage assessment",
            "summer": "Heat stress monitoring and ventilation check", "autumn": "Leaf and debris removal",
        },
    }


def make_7_day_forecast(forecast_data, model):
    """Generate 7-day solar output forecast"""
    try:
        forecast_df = pd.DataFrame(forecast_data["hourly"])
        forecast_df["time"] = pd.to_datetime(forecast_df["time"])
        forecast_df["hour"] = forecast_df["time"].dt.hour
        forecast_df["day_of_year"] = forecast_df["time"].dt.dayofyear
        forecast_df["month"] = forecast_df["time"].dt.month
        forecast_df["is_summer"] = forecast_df["month"].isin([6, 7, 8])
        forecast_df["is_peak_sun"] = (forecast_df["hour"] >= 10) & (forecast_df["hour"] <= 14)
        forecast_df["theoretical_panel_output"] = (forecast_df["global_tilted_irradiance"] * PANEL_AREA * PANEL_EFFICIENCY * SYSTEM_EFFICIENCY) / 1000
        daily_forecast = (
            forecast_df.groupby(forecast_df["time"].dt.date)
            .agg({"theoretical_panel_output": "sum", "temperature_2m": "mean", "cloud_cover": "mean", "precipitation": "sum"})
            .round(2)
        )
        return daily_forecast.to_dict("index")
    except Exception as e:
        logger.warning(f"Could not generate forecast: {e}")
        return {"error": "Forecast unavailable"}


def get_smart_solar_data(lat, lon, start_date_str, end_date_str, tilt, azimuth):
    """
    Fetches data by smartly splitting the request between the historical and
    forecast APIs based on the current date.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    today = datetime.now().date()
    
    historical_data = {'hourly': {}}
    forecast_data = {}
    
    if end_date.date() < today:
        historical_data = get_chunked_historical_data(lat, lon, start_date_str, end_date_str, tilt, azimuth)
    
    elif start_date.date() > today:
        forecast_response = get_forecast_data(lat, lon, tilt, azimuth)
        
        forecast_df = pd.DataFrame(forecast_response['hourly'])
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        
        filtered_forecast_df = forecast_df[
            (forecast_df['time'].dt.date >= start_date.date()) & 
            (forecast_df['time'].dt.date <= end_date.date())
        ]
        
        forecast_data = {
            'hourly': filtered_forecast_df.to_dict(orient='list'),
            'daily': forecast_response['daily'] 
        }

    else:
        historical_end_date = today - timedelta(days=1)
        historical_end_date_str = historical_end_date.strftime("%Y-%m-%d")
        historical_data = get_chunked_historical_data(lat, lon, start_date_str, historical_end_date_str, tilt, azimuth)
        
        forecast_response = get_forecast_data(lat, lon, tilt, azimuth)
        forecast_df = pd.DataFrame(forecast_response['hourly'])
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        
        filtered_forecast_df = forecast_df[
            (forecast_df['time'].dt.date >= today) & 
            (forecast_df['time'].dt.date <= end_date.date())
        ]
        
        forecast_data = {
            'hourly': filtered_forecast_df.to_dict(orient='list'),
            'daily': forecast_response['daily']
        }
    
    merged_data = historical_data.copy() if historical_data else {'hourly': {}}
    if forecast_data:
        if 'hourly' in merged_data:
            for key, values in forecast_data['hourly'].items():
                if key in merged_data['hourly']:
                    merged_data['hourly'][key].extend(values)
                else:
                    merged_data['hourly'][key] = values
        else:
            merged_data.update(forecast_data)
        
    return merged_data

def generate_perfect_solar_analysis(
    location_name=None,
    lat=None,
    lon=None,
    panel_area=20.0,
    current_tilt=35,
    current_azimuth=180,
    start_date=None,
    end_date=None,
):
    """
    Generate a complete solar analysis report based on lat/lon or a location name,
    with an optional historical date range.
    """
    global PANEL_AREA
    PANEL_AREA = panel_area

    if location_name:
        lat, lon = get_coordinates_from_location(location_name)
        if lat is None or lon is None:
            return {"error": "Invalid location name or geocoding service unavailable."}
    elif lat is None or lon is None:
        return {"error": "Either lat/lon or location_name must be provided."}
    
    if not start_date:
        start_date = "2023-01-01"
    if not end_date:
        end_date = "2023-12-31"

    try:
        # Step 1: Fetch all necessary data ONCE
        # Fetch data for the current configuration
        current_config_data = get_smart_solar_data(
            lat, lon, start_date, end_date, current_tilt, current_azimuth
        )
        
        # Fetch data for a horizontal setup (for optimization)
        horizontal_config_data = get_smart_solar_data(
            lat, lon, start_date, end_date, tilt=0, azimuth=0
        )
        
        # Step 2: Convert to DataFrames and perform analysis
        current_df = calculate_perfect_solar_features(current_config_data, lat, lon)
        horizontal_df = calculate_perfect_solar_features(horizontal_config_data, lat, lon)

        model = PerfectSolarPredictor()
        model.train(current_df)

        # Step 3: Optimization Analysis
        # Pass the pre-fetched dataframes directly
        optimization_result = find_perfect_panel_configuration(horizontal_df, lat, lon)["optimal_for_maximum_output"]
        
        current_output_sum = current_df["predicted_solar_output_kwh"].sum()
        optimal_output_sum = optimization_result["annual_output_kwh"]
        improvement_kwh = optimal_output_sum - current_output_sum
        improvement_percentage = (improvement_kwh / current_output_sum) * 100 if current_output_sum > 0 else 0
        annual_savings = improvement_kwh * ELECTRICITY_RATE
        
        impact_analysis = {
            "current_annual_output": current_output_sum,
            "optimal_annual_output": optimal_output_sum,
            "improvement_kwh": improvement_kwh,
            "improvement_percentage": improvement_percentage,
            "annual_financial_benefit": annual_savings,
            "optimal_tilt": optimization_result["tilt"],
            "optimal_azimuth": optimization_result["azimuth"],
            "payback_period_years": (INSTALLATION_COST / annual_savings if annual_savings > 0 else float("inf")),
        }

        # Step 4: Construct the final insights dictionary
        insights = {
            "location_analysis": {
                "latitude": lat, "longitude": lon, "solar_potential_rating": calculate_solar_potential_rating(current_df),
                "best_months": current_df.groupby("month")["predicted_solar_output_kwh"].mean().nlargest(3).index.tolist(),
                "worst_months": current_df.groupby("month")["predicted_solar_output_kwh"].mean().nsmallest(3).index.tolist(),
            },
            "current_system_performance": {
                "annual_output_kwh": current_output_sum, "monthly_average": current_output_sum / 12,
                "peak_daily_output": current_df.groupby(current_df["time"].dt.date)["predicted_solar_output_kwh"].sum().max(),
                "capacity_factor": (current_df["predicted_solar_output_kwh"].mean() / (panel_area * PANEL_EFFICIENCY)) * 100,
            },
            "optimization_recommendations": impact_analysis,
            "weather_impact_analysis": {
                "cloud_impact": calculate_weather_impact(current_df, "cloud_cover"), "temperature_impact": calculate_weather_impact(current_df, "temperature_2m"),
                "seasonal_variation": current_df.groupby("month")["predicted_solar_output_kwh"].sum().to_dict(),
            },
            "financial_projections": {
                "annual_savings": current_output_sum * ELECTRICITY_RATE,
                "25_year_savings": current_output_sum * ELECTRICITY_RATE * 25 * 0.98**25,
                "carbon_offset_tons_per_year": current_output_sum * 0.0004,
            },
            "maintenance_schedule": generate_maintenance_recommendations(current_df),
            "future_forecast": make_7_day_forecast(current_config_data, model),
            "raw_data": {
                "historical_hourly_output": current_df[['time', 'predicted_solar_output_kwh', 'temperature_2m', 'cloud_cover']].to_dict('records'),
                "historical_daily_output": current_df.groupby(current_df['time'].dt.date)['predicted_solar_output_kwh'].sum().to_dict(),
            }
        }

        return clean_json_output(insights)

    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user. Exiting gracefully.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

def main_cli_execution():
    """
    Main function to run the script from the command line for local testing.
    """
    analysis_results = generate_perfect_solar_analysis(location_name="Ahmedabad, Gujarat", panel_area=20.0, start_date="2024-01-01", end_date="2024-03-31")

    if analysis_results:
        print("\n" + "=" * 50)
        print("        COMPREHENSIVE SOLAR ANALYSIS REPORT")
        print("=" * 50 + "\n")

        if "error" in analysis_results:
            print(f"Error: {analysis_results['error']}")
            return

        loc = analysis_results["location_analysis"]
        perf = analysis_results["current_system_performance"]
        opt = analysis_results["optimization_recommendations"]
        fin = analysis_results["financial_projections"]

        print("--- 1. Executive Summary ---")
        print(f"Location: Latitude {loc['latitude']}, Longitude {loc['longitude']}")
        print(f"Solar Potential: {loc['solar_potential_rating']}")
        print(f"Estimated Annual Output: {perf['annual_output_kwh']:.2f} kWh")
        print(f"Estimated Annual Savings: ${fin['annual_savings']:.2f}")
        print(f"Optimal Annual Setup: Tilt {opt['optimal_tilt']}°, Azimuth {opt['optimal_azimuth']}°\n")

        print("--- 2. Current System Performance ---")
        print(f"Total Annual Production: {perf['annual_output_kwh']:.2f} kWh")
        print(f"Average Monthly Production: {perf['monthly_average']:.2f} kWh")
        print(f"Highest Daily Production: {perf['peak_daily_output']:.2f} kWh")
        print(f"System Capacity Factor: {perf['capacity_factor']:.2f}%\n")

        print("--- 3. Optimization and Financial Impact ---")
        print("Recommendation for Maximum Annual Output:")
        print(f"    > Tilt: {opt['optimal_tilt']}°, Azimuth: {opt['optimal_azimuth']}°")
        print(f"    > Improvement: +{opt['improvement_kwh']:.2f} kWh ({opt['improvement_percentage']:.2f}%) per year")
        print(f"    > Additional Annual Savings: ${opt['annual_financial_benefit']:.2f}\n")
        print(f"Payback Period for this optimization: {opt['payback_period_years']:.2f} years")

        print("\n--- 4. Weather Impact Analysis ---")
        weath = analysis_results["weather_impact_analysis"]
        print("Seasonal Output (kWh):")
        for month, total in weath["seasonal_variation"].items():
            print(f"    - Month {month}: {total:.2f}")
        print("\nCloud Cover Impact:")
        print(f"    - Correlation with output: {weath['cloud_impact']['correlation']:.2f}")
        print(f"    - Average output when very cloudy: {weath['cloud_impact']['average_reduction_high']:.2f} kWh")
        print(f"    - Average output when clear: {weath['cloud_impact']['average_reduction_low']:.2f} kWh\n")

        print("--- 5. Maintenance Recommendations ---")
        maint = analysis_results["maintenance_schedule"]
        print(f"Panel Cleaning Frequency: {maint['panel_cleaning']['frequency']}")
        print(f"Priority Months for Cleaning: {maint['panel_cleaning']['priority_months']}")
        print(f"Winter Task: {maint['seasonal_tasks']['winter']}\n")

        print("--- 6. 7-Day Solar Forecast ---")
        forecast = analysis_results["future_forecast"]
        if "error" in forecast:
            print(f"    > {forecast['error']}")
        else:
            for date, data in forecast.items():
                print(f"    > Date: {date}")
                print(f"      - Forecasted Daily Output: {data['theoretical_panel_output']:.2f} kWh")
                print(f"      - Avg. Temp: {data['temperature_2m']:.1f}°C, Avg. Cloud Cover: {data['cloud_cover']:.1f}%")
                print(f"      - Precipitation: {data['precipitation']:.2f} mm\n")
        print("=" * 50)
        print("Report complete. This is a model-based estimate.")
        print("=" * 50)


if __name__ == "__main__":
    main_cli_execution()