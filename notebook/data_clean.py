from pathlib import Path

import pandas as pd


def clean_csv(input_path: str | Path) -> pd.DataFrame:
    """Clean stock-price data and save a cleaned copy beside the input."""
    input_path = Path(input_path).expanduser().resolve()
    df = pd.read_csv(input_path)

    if "Date" not in df.columns and "Price" in df.columns:
        df = df.rename(columns={"Price": "Date"})

    if "Date" in df.columns:
        metadata_rows = df["Date"].astype(str).str.strip().isin({"Ticker", "Date"})
        df = df.loc[~metadata_rows].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

    for column in df.columns:
        if column != "Date":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()

    output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")
    df.to_csv(output_path, index=False)
    return df
