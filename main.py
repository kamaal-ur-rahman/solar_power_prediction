from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Solar Power Prediction API")

prediction_history = []

# Model paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

model = joblib.load(MODEL_DIR / "solar_power_model.joblib")
feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")


# Request models
class SolarInput(BaseModel):
    Time: str
    GHI: float = Field(ge=0)
    temp: float
    pressure: float = Field(gt=0)
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    rain_1h: float = Field(ge=0)
    clouds_all: float = Field(ge=0, le=100)
    weather_type: int = Field(ge=1, le=5)


class AnalyzeInput(SolarInput):
    actual_power_W: float = Field(ge=0)


class BatchInput(BaseModel):
    records: list[SolarInput] = Field(min_length=1, max_length=500)


# Prepare input
def prepare_input(data: SolarInput):
    time = pd.to_datetime(data.Time)

    df = pd.DataFrame([{
        "GHI": data.GHI,
        "temp": data.temp,
        "pressure": data.pressure,
        "humidity": data.humidity,
        "wind_speed": data.wind_speed,
        "rain_1h": data.rain_1h,
        "clouds_all": data.clouds_all,
        "weather_type": data.weather_type,
        "hour_sin": np.sin(2 * np.pi * time.hour / 24),
        "hour_cos": np.cos(2 * np.pi * time.hour / 24),
        "day_of_year_sin": np.sin(2 * np.pi * time.dayofyear / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * time.dayofyear / 365.25),
        "day_of_week": time.dayofweek
    }])

    df = pd.get_dummies(df, columns=["weather_type"], dtype=int)
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


def get_prediction(data: SolarInput):
    features = prepare_input(data)
    prediction = model.predict(features)[0]
    return max(0, float(prediction))


@app.get("/")
def home():
    return {
        "message": "Welcome, Operator! The Solar Power Prediction System is up and running, ready for power predictions and performance analysis."
    }


# Individual prediction
@app.post("/predict")
def predict(data: SolarInput):
    try:
        prediction = get_prediction(data)

        result = {
            "type": "Prediction",
            "timestamp": data.Time,
            "predicted_power_W": round(prediction, 2)
        }

        prediction_history.append(result)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Retrieve all stored results
@app.get("/predictions")
def get_all_predictions():
    return {
        "total_predictions": len(prediction_history),
        "predictions": prediction_history
    }


# Performance deviation analysis
@app.post("/analyze")
def analyze(data: AnalyzeInput):
    try:
        predicted = get_prediction(data)
        actual = data.actual_power_W

        deviation = actual - predicted
        absolute_deviation = abs(deviation)

        percentage = (
            round((absolute_deviation / predicted) * 100, 2)
            if predicted > 1 else None
        )

        threshold = 2770.72

        result = {
            "type": "Analysis",
            "timestamp": data.Time,
            "actual_power_W": actual,
            "predicted_power_W": round(predicted, 2),
            "deviation_W": round(deviation, 2),
            "absolute_deviation_W": round(absolute_deviation, 2),
            "deviation_percentage": percentage,
            "status": "High Deviation" if absolute_deviation > threshold else "Normal"
        }

        prediction_history.append(result)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Batch prediction
@app.post("/batch-predict")
def batch_predict(data: BatchInput):
    try:
        results = []

        for record in data.records:
            prediction = get_prediction(record)

            results.append({
                "type": "Batch Prediction",
                "timestamp": record.Time,
                "predicted_power_W": round(prediction, 2)
            })

        # Store all results after successful batch processing
        prediction_history.extend(results)

        return {
            "total_records": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/key")
def get_input_key():
    return {
        "title": "Solar Prediction API - Input Reference",

        "instructions": "Provide weather observations for the timestamp at which solar power is to be predicted.",

        "input_fields": {
            "Time": {
                "meaning": "Date and time of observation",
                "format": "YYYY-MM-DD HH:MM:SS"
            },
            "GHI": {
                "meaning": "Global Horizontal Irradiance",
                "unit": "W/m²",
                "observed_max": 229.2
            },
            "temp": {
                "meaning": "Ambient temperature",
                "unit": "°C",
                "observed_examples": "4.5 to 32"
            },
            "pressure": {
                "meaning": "Atmospheric pressure",
                "unit": "hPa",
                "observed_examples": "983 to 1029"
            },
            "humidity": {
                "meaning": "Relative humidity",
                "unit": "%",
                "valid_range": "0 to 100"
            },
            "wind_speed": {
                "meaning": "Wind speed",
                "unit": "m/s",
                "observed_examples": "1.6 to 8.7"
            },
            "rain_1h": {
                "meaning": "Rainfall during the previous hour",
                "unit": "mm",
                "observed_examples": "0 to 0.19"
            },
            "clouds_all": {
                "meaning": "Cloud coverage",
                "unit": "%",
                "valid_range": "0 to 100"
            },
            "weather_type": {
                "meaning": "Encoded weather category",
                "valid_values": [1, 2, 3, 4, 5],
                "encoding": "Categorical values are converted into one-hot encoded columns before prediction.",
                "note": "Category 1 is the baseline. The exact names of the five weather categories are not documented in the available dataset information."
            }
        },

        "output": {
            "predicted_power_W": "Predicted solar power output in Watts"
        },

        "note": "Observed ranges are guidance, not strict input limits. Values outside them may still be accepted if they satisfy the API validation rules."
    }
