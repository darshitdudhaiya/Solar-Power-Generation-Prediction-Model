# Solar Power Generation Prediction API

This project provides a robust backend API for solar power generation analysis and prediction. Built with **Python** using **FastAPI**, it fetches real-time and historical weather data, trains machine learning models to predict solar output, and offers a comprehensive report with optimization and financial insights.

---

## 🚀 Features

- **Data Acquisition**  
  Fetches solar-relevant weather data from the **Open-Meteo API**, intelligently handling historical and forecast data.

- **Intelligent Data Handling**  
  Uses `requests` with a robust retry strategy for reliable API calls and **pandas** for efficient data manipulation and analysis.

- **Machine Learning Pipeline**  
  A multi-layered predictive model uses **scikit-learn** to provide accurate solar output predictions.

- **Imputation**  
  Employs `sklearn.impute.SimpleImputer` to handle missing data (NaN values) in the dataset, ensuring the models can be trained without errors.

- **Model Ensemble**  
  The core prediction model is a hybrid of:

  - Physics-based model
  - `SGDRegressor`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`  
    All combined by a **LinearRegression meta-model** for high accuracy.

- **Comprehensive Analysis**  
  Generates a detailed report including:

  - System performance metrics (annual output, capacity factor).
  - Optimization recommendations for panel tilt and azimuth.
  - Financial projections (annual savings, payback period).
  - Weather impact analysis and a 7-day solar forecast.

- **Robust API**  
  A **FastAPI** application provides a clear, well-documented API with endpoints for performing analysis and fetching reports.  
  Includes error handling and data validation using **Pydantic** and `HTTPException`.

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone [repository_url]
cd Solar_Power_Generation_Prediction_Model
```

### 2. Install dependencies

It’s recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

> **Note**: You will need to create a `requirements.txt` file from your project's dependencies.

### 3. Run the application

Navigate to the `solar_api` directory and start the server:

```bash
cd solar_api
uvicorn main:app --reload
```

The API will be accessible at:
👉 http://127.0.0.1:8000

Interactive docs will be available at:
👉 http://127.0.0.1:8000/docs

### 📡 API Usage

Make a POST request to the `/api/solar-analysis` endpoint with a JSON body.

- Endpoint: `http://127.0.0.1:8000/api/solar-analysis`
- Method: `POST`
- Content-Type: `application/json`

### Request Body

```json
{
  "location_name": "Ahmedabad ,Gujarat",
  "panel_area": 20.0,
  "current_tilt": 35.0,
  "current_azimuth": 180.0,
  "start_date": "2025-09-13",
  "end_date": "2025-09-30"
}
```
