import argparse
from pathlib import Path

import pandas as pd


# df = pd.open_csv("./Data/stocks_price.csv")
import yfinance as yf
from data_clean import clean_csv
from model import model_assignment


def download_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical stock prices from Yahoo Finance."""
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def train_stock_model(
    ticker: str,
    horizon: int = 1,
    period: str = "5y",
):
    df = download_stock_data(ticker,period)

    project_root = Path(__file__).resolve().parents[1]

    # 2. Locate project root
    project_root = Path(__file__).resolve().parents[1]

    # 3. Save raw data
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace(".", "_").replace("^", "")
    csv_path = data_dir / f"{safe_ticker}_stock.csv"

    df.to_csv(csv_path, index_label="Date")

    # 4. Clean downloaded data
    df = clean_csv(csv_path)

    print(f"{df}")

    # 6. Create dynamic future-price target
    df["Future_Close"] = df["Close"].shift(-horizon)

    df["Target"] = (
        df["Future_Close"] > df["Close"]
        ).astype(int)

    features = ["High", "Low", "Open", "Close", "Volume"]

    X = df[features]
    y = df["Target"]

    # Remove rows without future target data
    valid_rows = df["Future_Close"].notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # 7. Remove rows without future targets
    valid_targets = y.notna()

    X = X.loc[valid_targets]
    y = y.loc[valid_targets]

    # 8. Train model
    best_model, predictions = model_assignment(X, y)

    # 9. Save model using ticker and horizon
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        model_dir /
        f"{safe_ticker}_{horizon}days.pkl"
    )

    import joblib
    joblib.dump(best_model, model_path)

    return {
        "model": best_model,
        "model_path": str(model_path),
        "data": df,
        "features": features,
        "latest_data": df[features].iloc[[-1]],
        "ticker": ticker,
        "horizon": horizon,
        "predictions": predictions
    }
