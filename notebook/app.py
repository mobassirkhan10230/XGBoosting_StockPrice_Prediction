
import streamlit as st
import pandas as pd

from data_process import download_stock_data
from data_clean import clean_csv
from prediction import (
    predict_stock_direction,
    predict_future_close
)

from pathlib import Path


st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("Stock Market Prediction")
st.write("Train an XGBoost model and predict future stock prices.")


# Stock list
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "Axis Bank": "AXISBANK.NS"
}


# Initialize session state
if "trained" not in st.session_state:
    st.session_state.trained = False


# --------------------------------
# STAGE 1: MODEL TRAINING
# --------------------------------

st.header("1. Train Model")

selected_stock = st.selectbox(
    "Select a stock",
    list(STOCKS.keys())
)

ticker = STOCKS[selected_stock]

period = st.selectbox(
    "Select historical training period",
    ["1y", "2y", "5y", "10y"],
    index=2
)


if st.button("Train Model"):

    try:
        with st.spinner("Downloading data and training models..."):

            # Download historical stock data
            df = download_stock_data(
                ticker=ticker,
                period=period
            )

            if df.empty:
                raise ValueError("No stock data available.")

            # Flatten yfinance MultiIndex columns if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Remove invalid rows and clean data
            df = df.copy()

            required = ["High", "Low", "Open", "Close", "Volume"]
            df[required] = df[required].apply(
                pd.to_numeric,
                errors="coerce"
            )

            df = df.dropna(subset=required)

            # Keep downloaded data in session
            st.session_state.df = df
            st.session_state.ticker = ticker
            st.session_state.stock_name = selected_stock
            st.session_state.trained = True

        st.success("Historical data prepared successfully!")

    except Exception as e:
        st.session_state.trained = False
        st.error(f"Training failed: {e}")


# --------------------------------
# STAGE 2: STOCK PREDICTION
# --------------------------------

if st.session_state.trained:

    st.divider()

    st.header("2. Predict Stock")

    st.success(
        f"Data ready for {st.session_state.stock_name}"
    )

    latest_date = pd.to_datetime(
        st.session_state.df.index[-1]
    ).date()

    prediction_date = st.date_input(
        "Select future prediction date",
        value=latest_date + pd.Timedelta(days=7),
        min_value=latest_date + pd.Timedelta(days=1)
    )

    if st.button("Predict Stock"):

        try:
            with st.spinner("Training prediction models..."):

                # Calculate approximate business-day horizon.
                # Exchange holidays are not included in this first version.
                future_dates = pd.bdate_range(
                    start=latest_date + pd.Timedelta(days=1),
                    end=prediction_date
                )

                horizon = len(future_dates)

                if horizon < 1:
                    raise ValueError(
                        "Select a future trading date."
                    )

                df = st.session_state.df

                # Direction prediction
                direction = predict_stock_direction(
                    df,
                    horizon
                )

                # Price prediction
                price = predict_future_close(
                    df,
                    horizon
                )

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Closing Price",
                    f"₹{price['predicted_close']:.2f}"
                )

            with col2:
                st.metric(
                    "Stock Direction",
                    direction["prediction"]
                )

            st.write(
                f"UP probability: "
                f"{direction['up_probability']:.2%}"
            )

            st.write(
                f"Prediction date: {prediction_date}"
            )

            st.write(
                f"Trading-session horizon: {horizon}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
