# solar_api/analysis/models.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.impute import SimpleImputer  # Import the imputer
from solar_api.analysis.constants import PANEL_AREA, PANEL_EFFICIENCY, SYSTEM_EFFICIENCY


class PerfectSolarPredictor:
    def __init__(self):
        self.physics_model = self.create_physics_model()

        # Initialize an imputer for weather features
        self.weather_imputer = SimpleImputer(strategy="mean")
        self.weather_model = SGDRegressor()

        # Initialize an imputer for solar features
        self.solar_imputer = SimpleImputer(strategy="mean")
        self.temporal_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5
        )
        self.solar_model = GradientBoostingRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.12
        )
        self.meta_model = LinearRegression()

    def create_physics_model(self):
        def physics_predict(X):
            return (
                X["global_tilted_irradiance"]
                * PANEL_AREA
                * PANEL_EFFICIENCY
                * SYSTEM_EFFICIENCY
                / 1000
                * (1 + (X["temperature_2m"] - 25) * -0.004)
            )

        return physics_predict

    def prepare_features(self, df):
        weather_features = [
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "snow_depth",
            "pressure_msl",
        ]
        temporal_features = [
            "hour",
            "day_of_year",
            "month",
            "is_summer",
            "is_peak_sun",
            "sun_elevation",
            "air_mass",
        ]
        solar_features = [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "global_tilted_irradiance",
            "sunshine_duration",
        ]
        all_features = weather_features + temporal_features + solar_features
        return {
            "weather": df[weather_features],
            "temporal": df[temporal_features],
            "solar": df[solar_features],
            "all": df[all_features],
        }

    def train(self, df, target_column="predicted_solar_output_kwh"):
        # Drop rows where the target variable is NaN
        cleaned_df = df.dropna(subset=[target_column]).copy()

        # Check if any data is left after cleaning
        if cleaned_df.empty:
            raise ValueError(
                "Target variable contains no valid data points after cleaning."
            )

        features = self.prepare_features(cleaned_df)
        y = cleaned_df[target_column]

        # Impute missing values for weather features before training
        # fit_transform() learns the imputation strategy and applies it
        imputed_weather_features = self.weather_imputer.fit_transform(
            features["weather"]
        )
        self.weather_model.fit(imputed_weather_features, y)

        # Impute missing values for solar features before training
        imputed_solar_features = self.solar_imputer.fit_transform(features["solar"])
        self.temporal_model.fit(features["temporal"], y)
        self.solar_model.fit(imputed_solar_features, y)

        weather_pred = self.weather_model.predict(imputed_weather_features)
        temporal_pred = self.temporal_model.predict(features["temporal"])
        solar_pred = self.solar_model.predict(imputed_solar_features)
        physics_pred = cleaned_df.apply(self.physics_model, axis=1)

        meta_features = np.column_stack(
            [weather_pred, temporal_pred, solar_pred, physics_pred]
        )
        self.meta_model.fit(meta_features, y)

    def predict(self, df):
        features = self.prepare_features(df)

        # Apply the same imputation to new data for prediction
        imputed_weather_features = self.weather_imputer.transform(features["weather"])
        imputed_solar_features = self.solar_imputer.transform(features["solar"])

        weather_pred = self.weather_model.predict(imputed_weather_features)
        temporal_pred = self.temporal_model.predict(features["temporal"])
        solar_pred = self.solar_model.predict(imputed_solar_features)
        physics_pred = df.apply(self.physics_model, axis=1)

        meta_features = np.column_stack(
            [weather_pred, temporal_pred, solar_pred, physics_pred]
        )
        final_prediction = self.meta_model.predict(meta_features)

        return {
            "final_prediction": final_prediction,
            "weather_component": weather_pred,
            "temporal_component": temporal_pred,
            "solar_component": solar_pred,
            "physics_component": physics_pred,
        }
