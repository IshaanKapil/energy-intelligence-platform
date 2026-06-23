"""
fetch_real_data.py
------------------
Pulls REAL historical + live data. Two modes:

  python fetch_real_data.py           # historical training data (Kaggle + Open-Meteo archive)
  python fetch_real_data.py --live    # current + 7-day forecast weather (Open-Meteo forecast API)

Historical sources (free):
  1) LOAD  -> PJM AEP Hourly (Kaggle, free account + API token required)
  2) WEATHER/SOLAR -> Open-Meteo Historical Archive API (free, NO API KEY)

Live/forecast source (free, NO API KEY):
  3) Open-Meteo Forecast API -> current conditions + 7-day hourly forecast

Install deps:
    pip install pandas requests kagglehub

Kaggle auth (one-time): create a free account, go to Account -> "Create New
API Token", place kaggle.json at C:\\Users\\<you>\\.kaggle\\kaggle.json

Output:
  data/energy_dataset_real.csv   historical training data (load + weather + solar)
  data/weather_live.csv          live/forecast weather for real-time inference
"""

import argparse
from datetime import date, timedelta

import pandas as pd
import requests
from pathlib import Path

OUT = Path("data"); OUT.mkdir(exist_ok=True)

LAT, LON = 40.0, -82.9          # central Ohio (AEP territory)
SOLAR_CAPACITY_MW = 20.0        # hypothetical plant size

# Training window: PJM AEP dataset covers up to Aug 2018
# Open-Meteo archive lags ~5 days behind today
TRAIN_START = "2015-01-01"
TRAIN_END   = "2018-08-03"      # PJM AEP dataset ends here


# ---------------------------------------------------------------------------
# 1) LOAD — PJM hourly via Kaggle
# ---------------------------------------------------------------------------
def get_load(start=TRAIN_START, end=TRAIN_END):
    import kagglehub
    print("  Downloading PJM AEP hourly from Kaggle ...")
    path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    load = pd.read_csv(Path(path) / "AEP_hourly.csv")
    load.columns = ["datetime", "load_mw"]
    load["datetime"] = pd.to_datetime(load["datetime"])
    load = (load.drop_duplicates("datetime")
                .set_index("datetime").sort_index()
                .loc[start:end])
    print(f"  load rows: {len(load)}")
    return load


# ---------------------------------------------------------------------------
# 2) WEATHER (archive) — Open-Meteo Historical Archive (free, no key)
# ---------------------------------------------------------------------------
def get_weather_archive(start=TRAIN_START, end=TRAIN_END):
    print(f"  Fetching archived weather {start} -> {end} from Open-Meteo ...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
        "timezone": "America/New_York",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return _parse_weather(r.json()["hourly"])


# ---------------------------------------------------------------------------
# 3) WEATHER (live/forecast) — Open-Meteo Forecast API (free, no key)
#    Gives current conditions + 16 days ahead
# ---------------------------------------------------------------------------
def get_weather_live():
    print("  Fetching live + forecast weather from Open-Meteo ...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
        "timezone": "America/New_York",
        "forecast_days": 7,
        "past_days": 2,        # include last 2 days for context
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return _parse_weather(r.json()["hourly"])


def _parse_weather(h: dict) -> pd.DataFrame:
    w = pd.DataFrame(h)
    w["datetime"] = pd.to_datetime(w["time"])
    w = w.drop(columns="time").set_index("datetime").sort_index()
    w = w.rename(columns={
        "temperature_2m":     "temperature_c",
        "cloud_cover":        "cloud_cover_pct",
        "shortwave_radiation":"irradiance_wm2",
        "wind_speed_10m":     "wind_speed",
    })
    w["solar_mw"] = (w["irradiance_wm2"] / 1000.0 * SOLAR_CAPACITY_MW).clip(lower=0).round(3)
    return w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_historical():
    weather = get_weather_archive()
    load    = get_load()
    df = load.join(weather, how="inner").dropna()
    out = OUT / "energy_dataset_real.csv"
    df.to_csv(out)
    print(f"\nSaved {out}  rows={len(df)}")
    print(df.head())
    return df


def build_live():
    df = get_weather_live()
    out = OUT / "weather_live.csv"
    df.to_csv(out)
    print(f"\nSaved {out}  rows={len(df)}")
    print(df.head())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Fetch live/forecast weather (no Kaggle needed)")
    args = parser.parse_args()

    if args.live:
        build_live()
    else:
        build_historical()
        print("\nTip: run with --live to also fetch current + 7-day forecast weather.")
