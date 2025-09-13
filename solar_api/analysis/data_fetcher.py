# solar_api/analysis/data_fetcher.py

import requests
import logging
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    Timeout,
    RequestException,
)
from urllib3.util.retry import Retry
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime, timedelta, date

logger = logging.getLogger(__name__)


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
                response = session.get(
                    url, params=params, timeout=(10, 60), stream=False
                )
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
                logger.warning(
                    f"Connection/timeout error on attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    raise
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(
                        f"HTTP {e.response.status_code} error on attempt {attempt + 1}: {e}"
                    )
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
            logger.info(
                f"Successfully geocoded '{location_name}' to ({location.latitude}, {location.longitude})"
            )
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

    combined_data = {"hourly": {}}

    current_chunk_start = start_date
    while current_chunk_start <= end_date:
        current_chunk_end = (
            current_chunk_start.replace(day=28) + timedelta(days=4)
        ).replace(day=1) - timedelta(days=1)

        if current_chunk_end > end_date:
            current_chunk_end = end_date

        chunk_start_str = current_chunk_start.strftime("%Y-%m-%d")
        chunk_end_str = current_chunk_end.strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start_str,
            "end_date": chunk_end_str,
            "hourly": [
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "global_tilted_irradiance",
                "cloud_cover",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "precipitation",
                "snow_depth",
                "sunshine_duration",
                "pressure_msl",
                "vapour_pressure_deficit",
            ],
            "daily": [
                "sunrise",
                "sunset",
                "daylight_duration",
                "sunshine_duration",
                "shortwave_radiation_sum",
            ],
            "tilt": tilt,
            "azimuth": azimuth,
            "timezone": "auto",
        }

        try:
            chunk_data = make_robust_request(
                "https://archive-api.open-meteo.com/v1/archive", params=params
            )

            if chunk_data and "hourly" in chunk_data and not combined_data["hourly"]:
                combined_data = chunk_data
            elif chunk_data and "hourly" in chunk_data:
                for key, values in chunk_data["hourly"].items():
                    combined_data["hourly"][key].extend(values)
            else:
                logger.warning(
                    f"No data returned for chunk {chunk_start_str} to {chunk_end_str}. Skipping."
                )

        except RequestException as e:
            logger.error(
                f"Failed to fetch historical data for chunk {chunk_start_str} to {chunk_end_str}: {e}"
            )
            raise

        current_chunk_start = current_chunk_end + timedelta(days=1)

    return combined_data


def get_forecast_data(lat, lon, tilt=35, azimuth=180):
    """Fetches the next 7 days of solar-relevant data from Open-Meteo's forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "global_tilted_irradiance",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "snow_depth",
            "sunshine_duration",
            "pressure_msl",
            "vapour_pressure_deficit",
        ],
        "daily": [
            "sunrise",
            "sunset",
            "daylight_duration",
            "sunshine_duration",
            "shortwave_radiation_sum",
        ],
        "tilt": tilt,
        "azimuth": azimuth,
        "timezone": "auto",
    }
    return make_robust_request("https://api.open-meteo.com/v1/forecast", params=params)

    """Fetches the next 7 days of solar-relevant data from Open-Meteo's forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "global_tilted_irradiance",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "snow_depth",
            "sunshine_duration",
            "pressure_msl",
            "vapour_pressure_deficit",
        ],
        "daily": [
            "sunrise",
            "sunset",
            "daylight_duration",
            "sunshine_duration",
            "shortwave_radiation_sum",
        ],
        "tilt": tilt,
        "azimuth": azimuth,
        "timezone": "auto",
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

    historical_data = {"hourly": {}}
    forecast_data = {}

    if end_date.date() < today:
        historical_data = get_chunked_historical_data(
            lat, lon, start_date_str, end_date_str, tilt, azimuth
        )

    elif start_date.date() > today:
        forecast_response = get_forecast_data(lat, lon, tilt, azimuth)

        forecast_df = pd.DataFrame(forecast_response["hourly"])
        forecast_df["time"] = pd.to_datetime(forecast_df["time"])

        filtered_forecast_df = forecast_df[
            (forecast_df["time"].dt.date >= start_date.date())
            & (forecast_df["time"].dt.date <= end_date.date())
        ]

        forecast_data = {
            "hourly": filtered_forecast_df.to_dict(orient="list"),
            "daily": forecast_response["daily"],
        }

    else:
        historical_end_date = today - timedelta(days=1)
        historical_end_date_str = historical_end_date.strftime("%Y-%m-%d")
        historical_data = get_chunked_historical_data(
            lat, lon, start_date_str, historical_end_date_str, tilt, azimuth
        )

        forecast_response = get_forecast_data(lat, lon, tilt, azimuth)
        forecast_df = pd.DataFrame(forecast_response["hourly"])
        forecast_df["time"] = pd.to_datetime(forecast_df["time"])

        filtered_forecast_df = forecast_df[
            (forecast_df["time"].dt.date >= today)
            & (forecast_df["time"].dt.date <= end_date.date())
        ]

        forecast_data = {
            "hourly": filtered_forecast_df.to_dict(orient="list"),
            "daily": forecast_response["daily"],
        }

    merged_data = historical_data.copy() if historical_data else {"hourly": {}}
    if forecast_data:
        if "hourly" in merged_data:
            for key, values in forecast_data["hourly"].items():
                if key in merged_data["hourly"]:
                    merged_data["hourly"][key].extend(values)
                else:
                    merged_data["hourly"][key] = values
        else:
            merged_data.update(forecast_data)

    return merged_data


def generate_maintenance_recommendations(df):
    """Generate maintenance schedule based on weather patterns"""
    return {
        "panel_cleaning": {
            "frequency": "Monthly",
            "priority_months": df.groupby("month")["precipitation"]
            .mean()
            .nsmallest(3)
            .index.tolist(),
            "reason": "Low precipitation months require more frequent cleaning",
        },
        "inspection_schedule": {
            "quarterly_check": "Visual inspection and performance monitoring",
            "annual_check": "Professional inspection and electrical testing",
            "weather_related": "Check after severe weather events",
        },
        "seasonal_tasks": {
            "winter": "Snow removal and ice prevention",
            "spring": "Post-winter damage assessment",
            "summer": "Heat stress monitoring and ventilation check",
            "autumn": "Leaf and debris removal",
        },
    }

