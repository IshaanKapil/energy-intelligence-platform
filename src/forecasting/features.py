"""
features.py — shared feature engineering for forecasting models.

DESIGN NOTE (say this in interviews):
For a *day-ahead* forecast you cannot use load from the last few hours,
because at 9 AM today you are predicting all 24 hours of tomorrow — the most
recent value you truly have for tomorrow 2 PM is *today* 2 PM (lag 24) or
earlier. So every lag/rolling feature here uses a horizon of >= 24 hours.
This avoids look-ahead leakage and keeps the model honestly deployable.
"""

import numpy as np
import pandas as pd

HORIZON = 24  # forecast horizon in hours (day-ahead)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features (all known in advance -> safe to use)."""
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # cyclical encodings so the model knows 23:00 is next to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)
    return df


def add_load_lags(df: pd.DataFrame, target="load_mw") -> pd.DataFrame:
    """Lag/rolling features for load, all with horizon >= 24h (no leakage)."""
    df[f"{target}_lag24"] = df[target].shift(24)
    df[f"{target}_lag48"] = df[target].shift(48)
    df[f"{target}_lag168"] = df[target].shift(168)            # same hour last week
    # rolling stats computed on data available >= 24h ago
    df[f"{target}_roll24"] = df[target].shift(24).rolling(24).mean()
    df[f"{target}_roll168"] = df[target].shift(24).rolling(168).mean()
    return df


def build_load_features(df: pd.DataFrame):
    """Return (X, y) for the load model."""
    df = df.copy()
    df = add_calendar(df)
    df = add_load_lags(df, "load_mw")
    # weather is treated as a known forecast input (standard assumption)
    feature_cols = [
        "hour", "dayofweek", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "temperature_c",
        "load_mw_lag24", "load_mw_lag48", "load_mw_lag168",
        "load_mw_roll24", "load_mw_roll168",
    ]
    df = df.dropna(subset=feature_cols + ["load_mw"])
    return df[feature_cols], df["load_mw"], df.index


def build_solar_features(df: pd.DataFrame):
    """
    Return (X, y) for the solar model.
    We deliberately predict solar from WEATHER forecast inputs (cloud, temp,
    time-of-day, season) rather than from irradiance directly — that mirrors
    the real task of forecasting generation from a weather forecast.
    """
    df = df.copy()
    df = add_calendar(df)
    feature_cols = [
        "hour", "month",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "cloud_cover_pct", "temperature_c",
    ]
    df = df.dropna(subset=feature_cols + ["solar_mw"])
    return df[feature_cols], df["solar_mw"], df.index


def time_split(index, test_frac=0.2):
    """Chronological split — NEVER shuffle time series."""
    n = len(index)
    cut = int(n * (1 - test_frac))
    return slice(0, cut), slice(cut, n)
