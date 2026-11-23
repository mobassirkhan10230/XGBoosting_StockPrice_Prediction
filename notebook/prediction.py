
import numpy as np
import pandas as pd

from model import (
    train_direction_model,
    train_price_model
)

# Features used by the model
FEATURES = [
    "High",
    "Low",
    "Open",
    "Close",
    "Volume"
]


def prepare_training_data(df, horizon):

    df = df.copy()

    # Validate prediction horizon
    if horizon < 1:
        raise ValueError("Prediction horizon must be at least 1.")

    # Ensure data is sorted chronologically
    df = df.sort_index()

    # Convert features into numeric values
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinity with NaN
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Create future closing price
    df["Future_Close"] = df["Close"].shift(-horizon)

    # Create classification target
    # 1 = UP
    # 0 = NOT UP (DOWN or unchanged)
    df["Target"] = (
        df["Future_Close"] > df["Close"]
    ).astype(int)

    # Remove rows without a future closing price
    df = df.dropna(subset=["Future_Close"])

    # Remove rows containing invalid feature values
    df = df.dropna(subset=FEATURES)

    X = df[FEATURES].copy()
    y = df["Target"].copy()

    if len(X) < 100:
        raise ValueError(
            "Not enough valid historical data to train the model."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Training data contains only one target class. "
            "Cannot train a binary classifier."
        )

    return X, y


def prepare_price_training_data(df, horizon):

    df = df.copy()

    if horizon < 1:
        raise ValueError("Prediction horizon must be at least 1.")

    df = df.sort_index()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    df["Future_Close"] = df["Close"].shift(-horizon)
    df = df.dropna(subset=["Future_Close", *FEATURES])

    X = df[FEATURES].copy()
    y = df["Future_Close"].copy()

    if len(X) < 100:
        raise ValueError("Not enough valid historical data to train the model.")

    return X, y


def predict_stock_direction(df, horizon):

    # Prepare training data
    X, y = prepare_training_data(df, horizon)

    # Train a new classification model
    model = train_direction_model(X, y)

    # Prepare the latest available stock data
    latest_data = df.copy()

    latest_data[FEATURES] = latest_data[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    latest_data[FEATURES] = latest_data[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    latest_data = latest_data.dropna(subset=FEATURES)

    if latest_data.empty:
        raise ValueError("No valid latest stock data available.")

    latest_data = latest_data[FEATURES].iloc[[-1]]

    # Predict stock direction
    prediction = model.predict(latest_data)[0]

    # Calculate prediction probabilities
    probabilities = model.predict_proba(latest_data)[0]

    up_index = list(model.classes_).index(1)
    not_up_index = list(model.classes_).index(0)

    up_probability = probabilities[up_index]
    not_up_probability = probabilities[not_up_index]

    result = {
        "prediction": "UP" if prediction == 1 else "NOT UP",
        "up_probability": float(up_probability),
        "not_up_probability": float(not_up_probability),
        "horizon": horizon,
        "model": model
    }

    return result


def predict_future_close(df, horizon):

    if horizon < 1:
        raise ValueError("Horizon must be at least 1.")

    # Prepare regression training data
    X, y = prepare_price_training_data(df, horizon)

    # Train regression model
    model = train_price_model(X, y, horizon)

    # Prepare latest stock data
    latest_data = df.copy()

    latest_data[FEATURES] = latest_data[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    latest_data = latest_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    latest_data = latest_data.dropna(
        subset=FEATURES
    )

    if latest_data.empty:
        raise ValueError("No valid latest stock data.")

    latest_data = latest_data[FEATURES].iloc[[-1]]

    # Predict future closing price
    predicted_price = model.predict(latest_data)[0]

    return {
        "predicted_close": float(predicted_price),
        "current_close": float(latest_data["Close"].iloc[0]),
        "horizon": horizon,
        "model": model
    }

